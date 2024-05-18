from splitfolders import fixed

path = 'ASL_Dataset/'
fixed(path, output='ASL_Dataset_Reduced', seed=1337, fixed=(900, 300, 300))