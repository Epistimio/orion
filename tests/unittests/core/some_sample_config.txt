yo: 5
training/lr0~loguniform(0.0001, 0.3)
training/mbs~uniform(32, 256, discrete=True)

# some comments

layers/0/width=64
layers/0/type=relu
layers/1/width~uniform(32, 128, discrete=True)
layers/1/type~choices('relu', 'sigmoid', 'selu', 'leaky')
layers/2/width=16
layers/2/type~choices('relu', 'sigmoid', 'selu', 'leaky')

something-same~choices([1, 2, 3, 4, 5])
