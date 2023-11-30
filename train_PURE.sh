####
# Train the PURE model on LegalNER dataset
####

python PURE/run_entity.py \
    --do_train --do_eval \
    --learning_rate=1e-5  \
    --task_learning_rate=5e-4 \
    --train_batch_size=16 \
    --context_window 0 \
    --task LegalNER \
    --data_dir Data/PURE_data \
    --model bert-base-uncased \
    --output_dir results/PURE_bert