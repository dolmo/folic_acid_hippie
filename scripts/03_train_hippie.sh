#!/usr/bin/env bash
# =============================================================================
# Step 1: Train HIPPIE (pretrain on public data, embed FA_T4) — 3-epoch sweep
# =============================================================================
# Usage:
#   bash scripts/03_train_hippie.sh 10       # single run at 10 epochs
#   bash scripts/03_train_hippie.sh --all    # run 1, 10, 20 epochs
#
# The script uses train_FA_analysis.py which is a copy of train_multimodal.py
# with FA_T4 added to dataset_files.
#
# Key choices:
#   --config full_model       : full cVAE (regularisation + augmentation + conditional encoder)
#   --finetune-without-labels False : pretrain-only, NO finetuning step (embeds ALL FA_T4 units)
#   --pretrain-max-epochs N   : sweep 1 / 10 / 20 to observe embedding collapse
#   --z_dim 32                : latent dimension (matches paper default)
#   --beta 4                  : KL weight
#   --dataset FA_T4           : dataset to embed (excluded from pretraining)
# =============================================================================

# Paths — override via environment variables if running on a different machine
VENV_PY=${HIPPIE_VENV_PYTHON:-/home/jesus/hippie_rebuttals/hip-hip-hippie/hippie_venv/bin/python}
HIPPIE_DIR=${HIPPIE_CODE_DIR:-/home/jesus/hippie_rebuttals/hip-hip-hippie}

run_one() {
    local EP=$1
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  HIPPIE training: FA_T4  |  config=full_model  |  epochs=${EP}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    cd "$HIPPIE_DIR"        
    $VENV_PY train_FA_analysis.py \
        --dataset FA_T4 \
        --config full_model \
        --pretrain-max-epochs "${EP}" \
        --finetune-max-epochs 0 \
        --finetune-without-labels False \
        --z_dim 32 \
        --beta 4 \
        --batch-size 512 \
        --learning-rate 0.001 \
        --weight-decay 0.01 \
        --acg-weight 1.0 \
        --wandb-tag "FA_T4_analysis" \
        2>&1 | tee "/tmp/hippie_FA_T4_epochs${EP}.log"
    echo "Done: epochs=${EP}"
}

if [ "$1" == "--all" ]; then
    for EP in 1 10 20; do
        run_one "$EP"
    done
else
    EP=${1:-10}
    run_one "$EP"
fi

echo ""
echo "Embeddings saved to:"
echo "  /home/jesus/results_hippie_rebuttals/FA_T4/"
echo ""
echo "Next: run scripts/04_embedding_analysis.py to generate UMAP / classifier / attribution plots."