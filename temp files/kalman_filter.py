from filterpy.kalman import KalmanFilter

# Get these two from camera for initial position
x_position = 0
y_position = 0

# Kalman filter for x position
kf_x = KalmanFilter(dim_x=2, dim_z=1, dim_u=1)
kf_x.x = np.array([x_position, 0.])

kf_x.P = [[1,0],[0,1]]
kf_x.Q = [[1,0],[0,1]]
kf_x.R = [[1]]
kf_x.H = np.array([[1.,0.]])

kf_y = KalmanFilter(dim_x=2, dim_z=1, dim_u=1)
kf_y.x = np.array([y_position, 0.])

kf_y.P = [[1,0],[0,1]]
kf_y.Q = [[1,0],[0,1]]
kf_y.R = [[1]]
kf_y.H = np.array([[1.,0.]])

while True:
    
    #get data
    acc_x, acc_y,del_t = get_from_udp()
    position_x, position_y = get_from_camera()
    
    #update matrices
    kf_x.F = np.array([[1.,del_t],
                  [0.,1.]])
    
    kf_x.B = np.array([0.5*(del_t**2), del_t])
    
    #predict new values
    kf_x.predict(acc_x)
    kf_x.update(position_x)
    
    #update matrices
    kf_y.F = np.array([[1.,del_t],
                  [0.,1.]])
    kf_y.B = np.array([0.5*(del_t**2), del_t])
    
    #predict new values
    kf_y.predict(acc_y)
    kf_y.update(position_y)
    
    filtered_x = kf_x.x[0]
    filtered_y = kf.y.x[0]
    
    
