class GlobalVariables :

    #options for running different experiments
    use_samples = 1
    use_pitch = 0
    use_spectrogram = 0
    use_raw_data = 0

    #use_dense=1
    #use_CNN=0
    #Grid Size

    nRow = 3
    nCol = 3

    #parameters
    sample_state_size = 100
    pitch_state_size= 114
    spectrogram_length=129
    spectrogram_state_size= 259
    raw_data_state_size= 100
    action_size = 4
    batch_size = 32
    Number_of_episodes=100
    timesteps=50#50#(nRow+nCol+nRow)
    how_many_times = 3 #How many times to run the same experiment

