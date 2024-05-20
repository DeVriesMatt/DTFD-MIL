for i in {0..4};
do python olga_new_main.py --name "brca_${i}" --mDATA0_dir_train0 "/mnt/nvme0n1/ICCV/brca/features_train_${i}.pickle" --log_dir "/mnt/nvme0n1/ICCV/brca/logs_${i}" --mDATA0_dir_val0 "/mnt/nvme0n1/ICCV/brca/features_val_${i}.pickle" --mDATA_dir_test0 "/mnt/nvme0n1/ICCV/brca/features_test_${i}.pickle" --save_model_path "/mnt/nvme0n1/ICCV/brca/best_model_${i}.pth";
done