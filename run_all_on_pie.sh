# === run on PIE datasets ===

# benchmark comparison
python train_test.py -c config_files_pie/baseline/PCPA_jaad.yaml  # PCPA
python train_test.py -c config_files_pie/baseline/SingleRNN.yaml  # SingleRNN
python train_test.py -c config_files_pie/baseline/SFRNN.yaml      # SF-GRU
python train_test.py -c config_files_pie/ours/MASK_PCPA_jaad_2d.yaml   # ours

# ablation study
python train_test.py -c config_files_pie/laterfusion/MASK_PCPA_jaad.yaml    # ours1
python train_test.py -c config_files_pie/earlyfusion/MASK_PCPA_jaad.yaml  # ours2
python train_test.py -c config_files_pie/hierfusion/MASK_PCPA_jaad.yaml  # ours3
python train_test.py -c config_files_pie/baseline/PCPA_jaad_2d.yaml      # ours4
python train_test.py -c config_files_pie/laterfusion/MASK_PCPA_jaad_2d.yaml  # ours5
python train_test.py -c config_files_pie/earlyfusion/MASK_PCPA_jaad_2d.yaml  # ours6
python train_test.py -c config_files_pie/hierfusion/MASK_PCPA_jaad_2d.yaml  # ours7
