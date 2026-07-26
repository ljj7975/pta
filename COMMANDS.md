uvicorn gui.proto_viz_web.app:app --host 0.0.0.0 --port 8000 --reload

python -u runner.py --method patch_modulated_pta --config configs --datasets dtd --backbone ViT-B/16

python -u runner.py \
    --method patch_modulated_pta \
    --config configs/patch_modulated_pta \
    --clip-model clip_surgery \
    --datasets dtd \
    --backbone ViT-B/16


CMD=(python -u runner.py
    --method "$METHOD"
    --config "$CONFIG"
    --clip-model "$CLIP_MODEL"
    --datasets "$DATASET"
    --backbone ViT-B/16)
exp patch_modulated_pta configs/patch_modulated_pta   clip_surgery    ""                           "PatchModPTA-CLIPSurgery"


# Single image, augmented 6 times — no dataset needed:
python tests/debug_gaussian_patch.py \
    --image tests/assets/cat.jpg --class-name cat \
    --n-images 6 --filter-modes none cosine_with_labels

# Directory of same-class images + side-by-side filter mode comparison:
python tests/debug_gaussian_patch.py \
    --images-dir /share_98/datasets/public/caltech-101/101_ObjectCategories/Faces --class-name faces \
    --n-images 10 \
    --filter-modes none cosine_with_labels cosine_no_labels surgery_with_labels surgery_no_labels \
    --compare-modes


python tests/debug_gaussian_patch.py \
    --images-dir /share_98/datasets/public/caltech-101/101_ObjectCategories/Leopards --class-name leopards \
    --n-images 10 \
    --filter-modes none cosine_with_labels cosine_no_labels surgery_with_labels surgery_no_labels \
    --compare-modes

