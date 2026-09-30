// Copyright 2025 Jefferson Amstutz
// SPDX-License-Identifier: Apache-2.0

#include "Group.h"
#include "geometry/GeometryAttributes.h"
// cycles
#include "kernel/types.h"
#include "scene/object.h"
// std
#include <algorithm>

namespace anari_cycles {

Group::Group(CyclesGlobalState *s)
    : Object(ANARI_GROUP, s),
      m_surfaceData(this),
      m_volumeData(this),
      m_lightData(this)
{}

Group::~Group() = default;

void Group::commitParameters()
{
  m_surfaceData = getParamObject<ObjectArray>("surface");
  m_volumeData = getParamObject<ObjectArray>("volume");
  m_lightData = getParamObject<ObjectArray>("light");
}

bool Group::addGroupToCurrentCyclesScene(const math::mat4 &xfm,
    const helium::box1 &shutter,
    const std::vector<ccl::Transform> *motion,
    uint32_t instanceId,
    const InstanceAttributeValues *attributes) const
{
  auto &state = *deviceState();
  bool deformationMotion = false;

  auto cxfm = mat4ToCycles(xfm);

  // Object::set_motion() steals the array, so every object needs its own
  // copy of the baked motion steps.
  auto setMotion = [&](ccl::Object *o) {
    if (!motion || motion->size() < 2)
      return;
    ccl::array<ccl::Transform> steps;
    steps.resize(motion->size());
    std::copy(motion->begin(), motion->end(), steps.data());
    o->set_motion(steps);
  };

  // The instance 'id' feeds the 'instanceId' frame channel through a
  // per-object float attribute read by the OutputAOV nodes in every surface
  // material (see Material::makeGraph()). Stored biased by +1 -- exact in a
  // float for ids < 2^24 -- so the "no id" default (~0u) becomes 0, the
  // value untouched AOV pixels read back anyway. Volumes get the attribute
  // too, but Cycles never runs AOV nodes in volume shading, so volume
  // pixels always read back ~0u.
  const float instanceIdAttr = instanceId == ~0u ? 0.f : instanceId + 1.f;
  auto setInstanceId = [&](ccl::Object *o) {
    o->attributes.emplace_back(
        OIIO::ustring("instanceId"), OIIO::TypeFloat, 1, &instanceIdAttr);
  };

  // Instance attributes become per-object attributes, which the material's
  // AttributeNodes only read where the geometry lacks the attribute (Cycles
  // gives geometry attributes precedence, matching ANARI's lookup order).
  // 'color' always gets a value, because geometries without a color source
  // rely on the object-level default (see DEFAULT_COLOR).
  auto setInstanceAttributes = [&](ccl::Object *o) {
    for (int c = 0; c < Geometry::NUM_ATTRIBUTE_CHANNELS; c++) {
      std::optional<anari_vec::float4> v;
      if (attributes)
        v = (*attributes)[c];
      if (!v && c == CH_COLOR)
        v = DEFAULT_COLOR;
      if (!v)
        continue;
      const float value[4] = {(*v)[0], (*v)[1], (*v)[2], (*v)[3]};
      o->attributes.emplace_back(
          OIIO::ustring(CHANNEL_CYCLES_NAME[c]), ccl::TypeFloat4, 1, value);
    }
  };

  if (m_surfaceData) {
    auto **surfacesBegin = (Surface **)m_surfaceData->handlesBegin();
    auto **surfacesEnd = (Surface **)m_surfaceData->handlesEnd();

    std::for_each(surfacesBegin, surfacesEnd, [&](Surface *s) {
      if (!s->isValid()) {
        s->warnIfUnknownObject();
        return;
      }
      if (!s->cyclesGeometry())
        return;
      // KHR_GEOMETRY_*_MOTION_DEFORMATION: (re)bake the geometry's motion
      // keys onto this shutter (cached; no-op for non-deforming geometry).
      deformationMotion |= s->bakeGeometryMotion(shutter);
      auto *o = state.scene->create_node<ccl::Object>();
      o->set_geometry(s->cyclesGeometry());
      o->set_tfm(cxfm);
      setMotion(o);
      setInstanceId(o);
      setInstanceAttributes(o);
      o->set_pass_id(s->id());
      // Ray visibility / compositing flags live on the ANARI Surface, so
      // every instance of a surface shares them (documented limitation of
      // CYCLES_SURFACE_COMPOSITING). set_visibility() no-ops at the default
      // mask (~0u).
      o->set_visibility(s->visibilityMask());
      o->set_use_holdout(s->holdout());
      o->set_is_shadow_catcher(s->shadowCatcher());
      // CYCLES_LIGHT_LINKING receiver/blocker set indices and the
      // CYCLES_LIGHTGROUPS emission routing; all setters no-op at their
      // defaults (0 / 0 / empty).
      o->set_receiver_light_set(s->receiverLightSet());
      o->set_blocker_shadow_set(s->blockerShadowSet());
      o->set_lightgroup(OIIO::ustring(s->lightGroup()));
    });
  }

  if (m_volumeData) {
    auto **volumesBegin = (Volume **)m_volumeData->handlesBegin();
    auto **volumesEnd = (Volume **)m_volumeData->handlesEnd();

    std::for_each(volumesBegin, volumesEnd, [&](Volume *v) {
      if (!v->isValid()) {
        v->warnIfUnknownObject();
        return;
      }
      if (!v->cyclesGeometry())
        return;
      auto *o = state.scene->create_node<ccl::Object>();
      o->set_geometry(v->cyclesGeometry());
      o->set_tfm(cxfm);
      setMotion(o);
      setInstanceId(o);
      o->set_pass_id(v->id());
    });
  }

  if (m_lightData) {
    auto **lightsBegin = (Light **)m_lightData->handlesBegin();
    auto **lightsEnd = (Light **)m_lightData->handlesEnd();

    if (motion && motion->size() > 1 && lightsBegin != lightsEnd) {
      reportMessage(ANARI_SEVERITY_WARNING,
          "lights in a motion instance do not motion blur (unsupported by "
          "Cycles) -- placing them at the shutter-start pose");
    }

    std::for_each(lightsBegin, lightsEnd, [&](Light *l) {
      if (!l->isValid()) {
        l->warnIfUnknownObject();
        return;
      }
      // The light keeps track of its scene objects, so that later parameter
      // changes can be applied in place (see Light::finalize()).
      auto makeLightObject = [&](ccl::Light *cl, bool secondary) {
        auto *o = state.scene->create_node<ccl::Object>();
        o->set_geometry(cl);
        l->syncCyclesObject(o, xfm, secondary);
        l->addBakedObject(o, xfm, secondary);
      };
      makeLightObject(l->cyclesLight(), false);
      // Second emitter for e.g. two-sided quad lights.
      if (auto *second = l->secondaryCyclesLight())
        makeLightObject(second, true);
    });
  }

  return deformationMotion;
}

bool Group::hasDeformingSurfaces() const
{
  if (!m_surfaceData)
    return false;
  auto **surfacesBegin = (Surface **)m_surfaceData->handlesBegin();
  auto **surfacesEnd = (Surface **)m_surfaceData->handlesEnd();
  // Only surfaces that can render count (invalid ones are skipped by
  // addGroupToCurrentCyclesScene(), so their keys cannot depend on the
  // shutter).
  return std::any_of(surfacesBegin, surfacesEnd, [](const Surface *s) {
    return s->isValid() && s->geometry()->hasDeformationMotion();
  });
}

box3 Group::bounds() const
{
  box3 b = empty_box3();
  if (m_surfaceData) {
    auto **surfacesBegin = (Surface **)m_surfaceData->handlesBegin();
    auto **surfacesEnd = (Surface **)m_surfaceData->handlesEnd();

    std::for_each(surfacesBegin, surfacesEnd, [&](Surface *s) {
      if (s->isValid())
        extend(b, s->geometry()->bounds());
      else
        s->warnIfUnknownObject();
    });
  }
  if (m_volumeData) {
    auto **volumesBegin = (Volume **)m_volumeData->handlesBegin();
    auto **volumesEnd = (Volume **)m_volumeData->handlesEnd();

    std::for_each(volumesBegin, volumesEnd, [&](Volume *s) {
      if (s->isValid())
        extend(b, s->bounds());
      else
        s->warnIfUnknownObject();
    });
  }
  return b;
}

} // namespace anari_cycles

CYCLES_ANARI_TYPEFOR_DEFINITION(anari_cycles::Group *);
