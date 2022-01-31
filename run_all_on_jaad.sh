# === run on JAAD datasets ===

# benchmark comparison
python train_test.py -c config_files/baseline/PCPA_jaad.yaml  # PCPA
python train_test.py -c config_files/baseline/SingleRNN.yaml  # SingleRNN
python train_test.py -c config_files/baseline/SFRNN.yaml      # SF-GRU
python train_test.py -c config_files/ours/MASK_PCPA_jaad_2d.yaml  # ours

# ablation study
python train_test.py -c config_files/laterfusion/MASK_PCPA_jaad.yaml    # ours1
python train_test.py -c config_files/earlyfusion/MASK_PCPA_jaad.yaml  # ours2
python train_test.py -c config_files/hierfusion/MASK_PCPA_jaad.yaml  # ours3
python train_test.py -c config_files/baseline/PCPA_jaad_2d.yaml      # ours4
python train_test.py -c config_files/laterfusion/MASK_PCPA_jaad_2d.yaml  # ours5
python train_test.py -c config_files/earlyfusion/MASK_PCPA_jaad_2d.yaml  # ours6
python train_test.py -c config_files/hierfusion/MASK_PCPA_jaad_2d.yaml  # ours7
