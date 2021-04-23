from filterpy.kalman import KalmanFilter

# Get these two from camera for initial position
x_position = 0
y_position = 0
del_t = 2.0

# Kalman filter for x position
kf_x = KalmanFilter(dim_x=2, dim_z=1, dim_u=1)
kf_x.x = np.array([x_position, 0.])
kf_x.F = np.array([[1.,del_t],
                  [0.,1.]])
kf_x.P = ?
kf_x.Q = ?
kf_x.R = ?
kf_x.B = np.array([0.5*(del_t**2), del_t])
kf_x.H = np.array([[1.,0.]])

kf_y = KalmanFilter(dim_x=2, dim_z=1, dim_u=1)
kf_y.x = np.array([y_position, 0.])
kf_y.F = np.array([[1.,del_t],
                  [0.,1.]])
kf_y.P = ?
kf_y.Q = ?
kf_y.R = ?
kf_y.B = np.array([0.5*(del_t**2), del_t])
kf_y.H = np.array([[1.,0.]])

while True:
    acc_x, acc_y = get_from_udp()
    position_x, position_y = get_from_camera()
    
    kf_x.predict(acc_x)
    kf_x.update(position_x)
    
    kf_y.predict(acc_y)
    kf_y.update(position_y)
    
    kf_x.x[0] = filtered_x
    kf.y.x[0] = filtered_y
    
    mouse.move(filtered_x, filtered_y)
