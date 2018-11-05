
# coding: utf-8

# In[4]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import opensim as osim
from osim.env import ProstheticsEnv
from operator import add,sub


# In[5]:


class HiddenLayer:
    def __init__(self, M1, M2, name, f=tf.nn.relu, use_bias=True):
        self.W = tf.Variable(tf.random_normal(shape=(M1, M2)), name=name+'_W')
        self.params = [self.W]
        self.use_bias = use_bias
        if use_bias:
            self.b = tf.Variable(np.zeros(M2).astype(np.float32), name=name+'_b')
            self.params.append(self.b)
        self.f = f

    def forward(self, X):
        if self.use_bias:
            a = tf.matmul(X, self.W) + self.b
        else:
            a = tf.matmul(X, self.W)
        return self.f(a)


# In[12]:


class DQN:
    def __init__(self, D, K, hidden_layer_sizes, gamma, name, max_experiences=1500, min_experiences=150, batch_sz=30, env=None):
        self.K = K
        self.env = env
        
        # create the graph
        self.layers = []
        M1 = D
        for M2 in hidden_layer_sizes:
            layer = HiddenLayer(M1, M2, name)
            self.layers.append(layer)
            M1 = M2
        # final layer
        layer = HiddenLayer(M1, K, name, f=tf.nn.sigmoid)
        self.layers.append(layer)
        
        # collect params for copy
        self.params = []
        for layer in self.layers:
            self.params += layer.params
            
        # inputs and targets
        self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
        self.G = tf.placeholder(tf.float32, shape=(None,K), name='G')
        self.actions = tf.placeholder(tf.int32, shape=(None,K), name='actions')
        
        # calculate output and cost
        Z = self.X
        for layer in self.layers:
            Z = layer.forward(Z)
        Y_hat = Z
        self.predict_op = Y_hat
        
        #selected_action_values = tf.reduce_sum( Y_hat * tf.one_hot(self.actions, K), reduction_indices=[0])
        selected_action_values = Y_hat
        
        cost = tf.reduce_sum(tf.square(self.G - selected_action_values))
        self.train_op = tf.train.AdamOptimizer(1e-2).minimize(cost)
        
        # create replay memory
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences
        self.batch_sz = batch_sz
        self.gamma = gamma
    
        
    def set_session(self, session):
        self.session = session
    
    def copy_from(self, other):
        # collect all the ops
        ops = []
        my_params = self.params
        other_params = other.params
        for p, q in zip(my_params, other_params):
            actual = self.session.run(q)
            op = p.assign(actual)
            ops.append(op)
        # now run them all
        self.session.run(ops)
    
    def predict(self, X):
        X = np.atleast_2d(X)
        return self.session.run(self.predict_op, feed_dict={self.X: X})
    
    def train(self, target_network):
        # sample a random batch from buffer, do an iteration of GD
        if len(self.experience['s']) < self.min_experiences:
            # don't do anything if we don't have enough experience
            return
        
        # randomly select a batch
        idx = np.random.choice(len(self.experience['s']), size=self.batch_sz, replace=False)
        states = [self.experience['s'][i] for i in idx]
        actions = [self.experience['a'][i] for i in idx]
        rewards = [self.experience['r'][i] for i in idx]
        next_states = [self.experience['s2'][i] for i in idx]
        dones = [self.experience['done'][i] for i in idx]
        next_Q = target_network.predict(next_states)
        targets = [r*np.ones(len(self.gamma*next_q)) + self.gamma*next_q if not done else r for r, next_q, done in zip(rewards, next_Q, dones)]
        
        MIN_MAX = False
        states = np.array(states)
        targets = np.array(targets)
        # MIN-MAX Normalisation
        if MIN_MAX:
            state_mins = states.min(axis=0)
            state_maxs = states.max(axis=0)
            target_mins = targets.min(axis=0)
            target_maxs = targets.max(axis=0)
            states = (states - state_mins)/(state_maxs - state_mins + 0.00000001)
            targets = (targets - target_mins)/(target_maxs - target_mins + 0.00000001)
        # STD NORMAL Normalisation
        else:
            state_mean = states.mean(axis=0)
            state_std = states.std(axis=0) + 0.00000001
            target_mean = targets.mean(axis=0)
            target_std = targets.std(axis=0) + 0.00000001
            states = (states - state_mean)/state_std
            targets = (targets - target_mean)/target_std
        states = states.tolist()
        targets = targets.tolist()
        
        
        # call optimizer
        self.session.run(
          self.train_op,
          feed_dict={
            self.X: states,
            self.G: targets,
            self.actions: actions
          }
        )
        
    def add_experience(self, s, a, r, s2, done):
        if len(self.experience['s']) >= self.max_experiences:
            self.experience['s'].pop(0)
            self.experience['a'].pop(0)
            self.experience['r'].pop(0)
            self.experience['s2'].pop(0)
            self.experience['done'].pop(0)
        self.experience['s'].append(s)
        self.experience['a'].append(a)
        self.experience['r'].append(r)
        self.experience['s2'].append(s2)
        self.experience['done'].append(done)
        
    def sample_action(self, x, eps):
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            X = np.atleast_2d(x)
            return self.predict(X)[0]
        
    def save(self):
        # For saving
        tf.add_to_collection('vars', self.params)
        self.saver = tf.train.Saver()
        self.saver.save(self.session, './my_test_model')
        print("Saved")
        
    def restore(self):
        try:
            new_saver = tf.train.import_meta_graph('./my_test_model.meta')
#             new_saver = tf.train.Saver()
            new_saver.restore(self.session, tf.train.latest_checkpoint('./'))
            all_vars = tf.get_collection('vars')
            for param, old_param in zip(self.params,all_vars):
                param = old_param
            return 0
        except:
            print ("here")
            return 1


# In[13]:


def DictToList(state_desc):
    res = []
    pelvis = None
    pelvis_v_x = None
    headX_minus_pelvisX = None
    straight_legs = []
    leftfoot_rpros_z = None

    for body_part in ["pelvis", "head","torso","toes_l","pros_foot_r","talus_l","pros_tibia_r"]:
        cur = []
        cur += state_desc["body_pos"][body_part][0:2]
        cur += state_desc["body_vel"][body_part][0:2]
        cur += state_desc["body_acc"][body_part][0:2]
        cur += state_desc["body_pos_rot"][body_part][2:]
        cur += state_desc["body_vel_rot"][body_part][2:]
        cur += state_desc["body_acc_rot"][body_part][2:]
        if body_part == "pelvis":
            pelvis = cur
            res += cur[1:]
            pelvis_v_x = cur[3]
        else:
            if body_part == "head":
                headX_minus_pelvisX = cur[0] - pelvis[0]
            cur_upd = cur
            cur_upd[:2] = [cur[i] - pelvis[i] for i in range(2)]
            cur_upd[6:7] = [cur[i] - pelvis[i] for i in range(6,7)]
            res += cur_upd
    leftfoot_rpros_z = state_desc["body_pos"]["pros_foot_r"][2] - state_desc["body_pos"]["toes_l"][2]

    for joint in ["ankle_l","ankle_r","back","hip_l","hip_r","knee_l","knee_r"]:
        res += state_desc["joint_pos"][joint]
        res += state_desc["joint_vel"][joint]
        res += state_desc["joint_acc"][joint]
    straight_legs += state_desc["joint_pos"]["knee_l"] + state_desc["joint_pos"]["knee_r"]


    for muscle in sorted(state_desc["muscles"].keys()):
        res += [state_desc["muscles"][muscle]["activation"]]
        res += [state_desc["muscles"][muscle]["fiber_length"]]
        res += [state_desc["muscles"][muscle]["fiber_velocity"]]

    cm_pos = [state_desc["misc"]["mass_center_pos"][i] - pelvis[i] for i in range(2)]
    res = res + cm_pos + state_desc["misc"]["mass_center_vel"] + state_desc["misc"]["mass_center_acc"]
    cm_y = state_desc["misc"]["mass_center_pos"][1]

    return res, pelvis_v_x, headX_minus_pelvisX, cm_y, straight_legs, leftfoot_rpros_z


# In[17]:


def play_one(env, model, tmodel, eps, gamma, copy_period):
    observation = env.reset(project=False)
    observation,pv,hp,cm,legs,feets = DictToList(observation)
    done = False
    totalreward = 0.0
    iters = 0
    #prev_action = np.zeros(19)
    while not done and iters < 1000:
        action = model.sample_action(observation, eps)
        #if iters%10 == 0:
        #    print ("delta Action:", list( map(sub, prev_action, action)))
        #prev_action = action
        prev_observation = observation
        observation, reward, done, info = env.step(action, project=False)
        observation,pv,hp,cm,legs,feets = DictToList(observation)
            
        reward += pv * 0.01
        reward += 2
        reward += min(0,hp)*3 # head ahead of pelvis
        reward += min(0, cm - 0.85) * 2 #centre of mass adjustment
#         reward -= np.sum([max(0.0,k-0.1) for k in legs])*2 # straight legs penalty
        reward += feets * 4
        
        totalreward += reward
        if done and iters < 999:
            reward = -20
            break
        
        # update the model
        model.add_experience(prev_observation, action, reward, observation, done)
        model.train(tmodel)
        
        iters += 1
        
        if iters % copy_period == 0:
            tmodel.copy_from(model)
        
    return totalreward


# In[18]:


def plot_running_avg(totalrewards):
    N = len(totalrewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()


# In[19]:


env = ProstheticsEnv(visualize=True)
gamma = 0.9
copy_period = 50

D = len(DictToList(env.reset(project=False))[0])
# D = len(env.observation_space.sample())
K = len(env.action_space.sample())

hidden_layer_sizes = [80,60]
tf.reset_default_graph()
model = DQN(D, K, hidden_layer_sizes, gamma, "model", env=env)
tmodel = DQN(D, K, hidden_layer_sizes, gamma, "tmodel", env=env)
init = tf.global_variables_initializer()
session = tf.InteractiveSession()
session.run(init)
model.set_session(session)
tmodel.set_session(session)
if model.restore(): pass
if tmodel.restore():
    session.run(init)


# if 'monitor' in sys.argv:
#     filename = os.path.basename(__file__).split('.')[0]
#     monitor_dir = './' + filename + '_' + str(datetime.now())
#     env = wrappers.Monitor(env, monitor_dir)
    
N = 200
totalrewards = np.empty(N)
costs = np.empty(N)
for n in range(N):
    eps = 1.0/np.sqrt(n+1)
    totalreward = play_one(env, model, tmodel, eps, gamma, copy_period)
    totalrewards[n] = totalreward
    if n % 30 == 0:
        print("episode:", n, "total reward:", totalreward, "eps:", eps, "avg reward (last 100):", totalrewards[max(0, n-100):(n+1)].mean())

# tmodel.save("./t")
model.save()

session.close()
print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
print("total steps:", totalrewards.sum())

plt.plot(totalrewards)
plt.title("Rewards")
plt.show()

plot_running_avg(totalrewards)

