from importlib import import_module


def _safe_star_import(module_name):
    try:
        module = import_module(module_name, package=__name__)
        exported = getattr(module, '__all__', None)
        if exported is None:
            exported = [name for name in dir(module) if not name.startswith('_')]
        for name in exported:
            if hasattr(module, name):
                globals()[name] = getattr(module, name)
    except Exception:
        # Keep top-level package import resilient when optional deps are missing.
        pass


for _module_name in [
    '.color_transfer',
    '.naive_composition',
    '.opa_score',
    '.harmony_score',
    '.inharmonious_region_localization',
    '.image_harmonization',
    '.painterly_image_harmonization',
    '.fopa_heat_map',
    '.fos_score',
    '.kontext_blending_harmonization',
    '.shadow_generation',
    '.reflection_generation',
    '.os_insert',
]:
    _safe_star_import(_module_name)


__all__ = [
    'color_transfer', 'get_composite_image', 'OPAScoreModel',
    'HarmonyScoreModel', 'InharmoniousLocalizationModel',
    'ImageHarmonizationModel', 'PainterlyHarmonizationModel',
    'FOPAHeatMapModel', 'FOSScoreModel',
    'KontextBlendingHarmonizationModel',
    'ShadowGenerationModel', 'ReflectionGenerationModel',
    'OSInsertModel'
]
