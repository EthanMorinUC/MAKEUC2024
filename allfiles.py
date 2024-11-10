import cv2
from PIL import Image
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from threading import Thread
import numpy as np
import pickle
from objloader import *

INVERSE_MATRIX = np.array([[1.0, 1.0, 1.0, 1.0],
                           [-1.0, -1.0, -1.0, -1.0],
                           [-1.0, -1.0, -1.0, -1.0],
                           [1.0, 1.0, 1.0, 1.0]])

texture_id = 0
thread_quit = 0
X_AXIS = 0.0
Y_AXIS = 0.0
Z_AXIS = 0.0
DIRECTION = 1
current_view_matrix = np.array([])
new_frame = np.array([])

cap = cv2.VideoCapture(0)  # Use your camera index here
global mtx, dist
_, mtx, dist, _, _ = pickle.load(open("my_camera_calibration.p", "rb"))
new_frame = cap.read()[1]

def init():
    video_thread = Thread(target=update, args=())
    video_thread.start()

def init_gl(width, height):
    global texture_id
    global obj, obj2, obj3, obj4

    glClearColor(1.0, 1.0, 1.0, 1.0)
    glClearDepth(1.0)
    glDepthFunc(GL_LESS)
    glEnable(GL_DEPTH_TEST)
    glShadeModel(GL_SMOOTH)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, float(width) / float(height), 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)

    # Load objects
    obj = OBJ('EO0AAAMXQ0YGMC13XX7X56I3L_obj/EO0AAAMXQ0YGMC13XX7X56I3L.obj', swapyz=False)
    obj2 = OBJ('mario/HK40D7AZNBRT97PMCRL6ICE81.obj', swapyz=False)
    obj3 = OBJ("castle/castle.obj", swapyz=False)
    obj4 = OBJ("dragon/T582EI6OALSFRVGONWQ5YIKU2.obj", swapyz=False)

    # Assign texture
    glEnable(GL_TEXTURE_2D)
    texture_id = glGenTextures(1)

def track(frame):
    global mtx, dist, obj, obj2, obj3, obj4
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters_create()
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict,
                                              parameters=parameters,
                                              cameraMatrix=mtx,
                                              distCoeff=dist)
    if np.all(ids is not None):
        for i in range(0, len(ids)):
            if ids[i] == 1:
                rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 1, mtx, dist)
            elif ids[i] == 2:
                rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 1, mtx, dist)
            elif ids[i] == 3:
                rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 1, mtx, dist)
            else:
                rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.5, mtx, dist)
            rmtx = cv2.Rodrigues(rvec)[0]

            view_matrix = np.array([[rmtx[0][0], rmtx[0][1], rmtx[0][2], tvec[0, 0, 0]],
                                    [rmtx[1][0], rmtx[1][1], rmtx[1][2], tvec[0, 0, 1]],
                                    [rmtx[2][0], rmtx[2][1], rmtx[2][2], tvec[0, 0, 2]],
                                    [0.0, 0.0, 0.0, 1.0]])
            view_matrix = view_matrix * INVERSE_MATRIX
            view_matrix = np.transpose(view_matrix)
            glPushMatrix()
            glLoadMatrixd(view_matrix)
            glRotate(90, 1, 0, 0)
            glRotate(90, 0, 1, 0)
            glTranslate(0.5, 1.3, 0.2)
            if ids[i] == 1:
                obj2.render()
            elif ids[i] == 2:
                obj3.render()
            elif ids[i] == 3:
                obj4.render()
            else:
                obj.render()
            glPopMatrix()

    cv2.imshow('frame', frame)

def update():
    global new_frame
    while True:
        new_frame = cap.read()[1]
        if thread_quit == 1:
            break
    cap.release()
    cv2.destroyAllWindows()

def draw_gl_scene():
    global new_frame, texture_id

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)
    glLoadIdentity()
    frame = new_frame
    glDisable(GL_DEPTH_TEST)

    # Create texture
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, tx_image)
    
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glPushMatrix()
    glTranslatef(0.0, 0.0, -6.0)
    glBegin(GL_QUADS)
    glTexCoord2f(0.0, 1.0)
    glVertex3f(-4.0, -3.0, 0.0)
    glTexCoord2f(1.0, 1.0)
    glVertex3f(4.0, -3.0, 0.0)
    glTexCoord2f(1.0, 0.0)
    glVertex3f(4.0, 3.0, 0.0)
    glTexCoord2f(0.0, 0.0)
    glVertex3f(-4.0, 3.0, 0.0)
    glEnd()
    glPopMatrix()

    # Enable depth testing for 3D rendering
    glEnable(GL_DEPTH_TEST)

    # Track and render objects
    track(frame)

    glutSwapBuffers()

def key_pressed(key, x, y):
    global thread_quit
    key = key.decode("utf-8") 
    if key == chr(27) or key == 'q':
        thread_quit = 1
        sys.exit()

def run():
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(640, 480)
    glutInitWindowPosition(200, 200)
    window = glutCreateWindow('My AR Project')
    glutDisplayFunc(draw_gl_scene)
    glutIdleFunc(draw_gl_scene)
    glutKeyboardFunc(key_pressed)
    init_gl(640, 480)
    glutMainLoop()

if __name__ == "__main__":
    init()
    run()
