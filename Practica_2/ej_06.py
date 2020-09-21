



"""
# Identify the problem: 'XOR' or 'Image'
problem_flag = 'XOR' #P6
# Dataset
x_train = np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
y_train = np.array([[1],[-1],[-1],[1]])
# Regularizer
reg1 = regularizers.L2(0.1)
reg2 = regularizers.L1(0.2)
# Create model
model = models.Network()
model.add(layers.Dense(units=2,activations.Tanh(),input_dim=x_train.shape[1], regularizer=reg1))
model.add(layers.Dense(units=1,activations.Tanh(), regularizer=reg2))
# Train network
model.fit(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,
batch_size=x_train.shape[0], epochs=200, opt=optimizers.SGD(lr=0.05),
problem_flag=problem_flag)
"""