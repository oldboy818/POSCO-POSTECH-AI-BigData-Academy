done =False
while not done:
    ct2 = 0
    while not done:               
        action = env.action_space.sample()
        if env.available_act(action):
            break
        else:
            ct2 += 1
            if ct2 == 20:
                print ('ct2:20')
                done = True

    new_state, reward, done, info = env.step(action)
    env.render(action, reward=reward)