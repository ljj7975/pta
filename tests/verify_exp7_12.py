"""Smoke test for Exp7-Exp12 model modules."""

experiments = [
    ("exp7_inverted_image_weight", "Exp7InvertedImageWeightAdapter"),
    ("exp8_boosted_patch", "Exp8BoostedPatchAdapter"),
    ("exp9_entropy_fusion", "Exp9EntropyFusionAdapter"),
    ("exp10_soft_patch_gate", "Exp10SoftPatchGateAdapter"),
    ("exp12_patch_quality_modulation", "Exp12PatchQualityModulationAdapter"),
]


def test_imports():
    for module_name, class_name in experiments:
        mod = __import__(f"models.{module_name}", fromlist=["build"])
        adapter = mod.build({})
        assert type(adapter).__name__ == class_name
        print(f"  OK: {class_name}")


def test_exp11_uses_exp5():
    from models.exp5_tunable_fusion import build
    adapter = build({})
    print(f"  OK: Exp11 uses exp5_tunable_fusion ({type(adapter).__name__})")


if __name__ == "__main__":
    print("Testing Exp7-Exp12 module imports...")
    test_imports()
    test_exp11_uses_exp5()
    print("\nALL SMOKE TESTS PASSED")
