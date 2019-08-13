from PyQt5 import QtCore, Qt, QtWidgets, QtOpenGL, QtGui
import os
import numpy as np
from ..pointcloud import pointcloud

this_dir = os.path.dirname(__file__)


class viewer(QtWidgets.QOpenGLWidget):
    shader_dir = os.path.dirname(__file__)
    vertex_shader_path = os.path.join(shader_dir, 'vertex_shader.glsl')
    geometry_shader_path = os.path.join(shader_dir, 'geometry_shader.glsl')
    fragment_shader_path = os.path.join(shader_dir, 'fragment_shader.glsl')
    with open(vertex_shader_path, 'r') as f:
        vertex_shader = (QtGui.QOpenGLShader.Vertex, f.read())
    with open(geometry_shader_path, 'r') as f:
        geometry_shader = (QtGui.QOpenGLShader.Geometry, f.read())
    with open(fragment_shader_path, 'r') as f:
        fragment_shader = (QtGui.QOpenGLShader.Fragment, f.read())

    def __init__(self, parent=None):
        super(viewer, self).__init__(parent)
        self.__keys = {}
        self.__last_pos = None

        self.point_count = None
        self.set_camera((0, 0, 0), (0, 0, 1), (0, -1, 0))
        self.set_perspective(30, 1, 0.0, 1000.0)

    def initializeGL(self):
        self.gl = self.context().versionFunctions()

        # Set clear color!
        self.gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        self.gl.glEnable(self.gl.GL_DEPTH_TEST)

        # Add shaders to the program
        self.program = QtGui.QOpenGLShaderProgram(self)
        self.program.addShaderFromSourceCode(*self.vertex_shader)
        self.program.addShaderFromSourceCode(*self.geometry_shader)
        self.program.addShaderFromSourceCode(*self.fragment_shader)
        # Link and bind program. Do not release until VAO is finished!
        self.program.link()
        self.program.bind()

        # Create and set buffers in current OpenGL context
        self.buffer = QtGui.QOpenGLBuffer()
        self.buffer.create()
        self.buffer.bind()
        self.buffer.setUsagePattern(QtGui.QOpenGLBuffer.StaticDraw)

        # Bind the array object. Do not release until attrib.locations are set
        self.array_object = QtGui.QOpenGLVertexArrayObject()
        self.array_object.create()
        self.array_object.bind()

        self.program.enableAttributeArray(0)
        self.program.enableAttributeArray(1)
        self.program.enableAttributeArray(2)
        self.program.setAttributeBuffer(0, self.gl.GL_FLOAT, 00, 3, 36)
        self.program.setAttributeBuffer(1, self.gl.GL_FLOAT, 12, 3, 36)
        self.program.setAttributeBuffer(2, self.gl.GL_FLOAT, 24, 3, 36)

        self.program.setUniformValue("radius", 0.1)

        self.array_object.release()
        self.program.release()

    def set_vertices(self, vertices):
        self.__new_vertices = vertices
        self.update()

    def set_camera(self, position=None, look_at=None, up=None):
        self.__new_view = [position, look_at, up]
        for i, v in enumerate(self.__new_view):
            if v is None:
                self.__new_view[i] = self.__view[i]
        self.__view = self.__new_view
        view = [Qt.QVector3D(*v) for v in self.__view]
        v = Qt.QMatrix4x4()
        v.lookAt(*view)
        self.__new_view = v
        self.view_matrix = np.array([[v[0, 0], v[0, 1], v[0, 2], v[0, 3]],
                                     [v[1, 0], v[1, 1], v[1, 2], v[1, 3]],
                                     [v[2, 0], v[2, 1], v[2, 2], v[2, 3]],
                                     [v[3, 0], v[3, 1], v[3, 2], v[3, 3]]])
        self.update()

    def set_perspective(self, vangle=None, aspect=None, near=None, far=None):
        self.__new_perspective = [vangle, aspect, near, far]
        for i, v in enumerate(self.__new_perspective):
            if v is None:
                self.__new_perspective[i] = self.__perspective[i]
        self.__perspective = self.__new_perspective
        p = Qt.QMatrix4x4()
        p.setToIdentity()
        p.perspective(*self.__perspective)
        self.__new_perspective = p
        self.persp_matrix = np.array([[p[0, 0], p[0, 1], p[0, 2], p[0, 3]],
                                      [p[1, 0], p[1, 1], p[1, 2], p[1, 3]],
                                      [p[2, 0], p[2, 1], p[2, 2], p[2, 3]],
                                      [p[3, 0], p[3, 1], p[3, 2], p[3, 3]]])
        self.update()

    def updateGLVertices(self, vertices):
        self.__new_vertices = None

        data = vertices.astype('f4').tostring()
        self.point_count = len(vertices)
        self.buffer.allocate(data, len(data))

    def updateGLView(self, view):
        self.__new_view = None
        self.program.setUniformValue("view_matrix", view)

    def updateGLPerspective(self, perspective):
        self.__new_perspective = None
        self.program.setUniformValue("perspective_matrix", perspective)

    def paintGL(self):
        # Qt.QCoreApplication.processEvents()

        self.gl.glClear(self.gl.GL_COLOR_BUFFER_BIT)
        self.gl.glClear(self.gl.GL_DEPTH_BUFFER_BIT)

        self.program.bind()
        self.array_object.bind()

        if self.__new_view is not None:
            self.updateGLView(self.__new_view)
        if self.__new_perspective is not None:
            self.updateGLPerspective(self.__new_perspective)
        if self.__new_vertices is not None:
            self.updateGLVertices(self.__new_vertices)
        if self.point_count is None:
            return

        self.gl.glDrawArrays(self.gl.GL_POINTS, 0, self.point_count)
        self.array_object.release()
        self.program.release()

    def resizeGL(self, width, height):
        self.set_perspective(aspect=width / height)

    mouseDragged = QtCore.pyqtSignal(np.ndarray, np.ndarray, list, dict)
    mouseWheeled = QtCore.pyqtSignal(int)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Control:
            self.__keys['control'] = True
            if self.__last_pos is None:
                self.setCursor(QtCore.Qt.OpenHandCursor)
            else:
                self.setCursor(QtCore.Qt.ClosedHandCursor)

    def keyReleaseEvent(self, event):
        if event.key() == QtCore.Qt.Key_Control:
            self.__keys['control'] = False
            self.unsetCursor()

    def mousePressEvent(self, event):
        self.__last_pos = np.array([event.x(), event.y()])
        if 'control' in self.__keys and self.__keys['control']:
            self.setCursor(QtCore.Qt.ClosedHandCursor)

    def mouseReleaseEvent(self, event):
        self.__last_pos = None
        if 'control' in self.__keys and self.__keys['control']:
            self.setCursor(QtCore.Qt.OpenHandCursor)

    def mouseMoveEvent(self, event):
        buttons = []
        if event.buttons() & QtCore.Qt.LeftButton:
            buttons.append("Left")
        if event.buttons() & QtCore.Qt.MidButton:
            buttons.append("Mid")
        if event.buttons() & QtCore.Qt.RightButton:
            buttons.append("Right")
        previous = self.__last_pos
        self.__last_pos = np.array([event.x(), event.y()])
        self.mouseDragged.emit(previous, self.__last_pos, buttons, self.__keys)

    def wheelEvent(self, event):
        self.mouseWheeled.emit(event.angleDelta().y())


class app():
    def __init__(self, argv=[]):
        format = Qt.QSurfaceFormat()
        format.setVersion(4, 1)
        format.setProfile(Qt.QSurfaceFormat.CoreProfile)
        format.setDepthBufferSize(32)  # 32 bits
        Qt.QSurfaceFormat.setDefaultFormat(format)
        self.__app = Qt.QApplication(argv)

        self.viewer = viewer()

        self.timer = Qt.QElapsedTimer()
        self.timer.start()

        self.viewer.mouseDragged.connect(self.transform)
        self.viewer.mouseWheeled.connect(self.zoom)

        self.center = np.array([89.21518974,  60.11755026, 628.59493419])
        norm = np.linalg.norm(self.center)
        self.zoom_scale = 300
        self.zoom_level = np.log(norm) * self.zoom_scale
        self.cam_vec = -(self.center / norm)
        self.up_vec = np.array([0., -1., 0.])
        self.update_view()
        self.update_perspective()

    def update_view(self):
        zoom = np.exp(self.zoom_level / self.zoom_scale)
        pos = self.center + zoom * self.cam_vec
        self.viewer.set_camera(pos, self.center, self.up_vec)

    def update_perspective(self):
        near = 0.2 * np.exp(self.zoom_level / self.zoom_scale)
        far = None  # 1.2 * np.exp(self.zoom_level / self.zoom_scale)
        self.viewer.set_perspective(near=near, far=far)

    def arcball(self, screen_position):
        screen = np.array([self.viewer.width(), self.viewer.height()])
        screen_position = (1.0 * screen_position / screen) * 2 - 1
        screen_position[1] *= -1
        dist2 = np.asscalar(screen_position @ screen_position)
        radius2 = 1**2
        if dist2 <= radius2:
            return np.array([*screen_position, np.sqrt(radius2 - dist2)])
        else:
            return np.array([*(screen_position / np.sqrt(dist2)), 0])

    def transform(self, previous, current, buttons, keys):
        if (previous == current).all():
            return
        if 'control' in keys and keys['control']:
            return self.translate(previous, current, buttons)
        else:
            return self.rotate(previous, current, buttons)

    def translate(self, previous, current, buttons):
        iV = np.linalg.inv(self.viewer.view_matrix[:3, :3])
        delta = np.array([*(0.5 * (previous - current)), 0])
        delta[1] *= -1
        self.center = self.center + iV @ delta
        self.update_view()

    def rotate(self, previous, current, buttons):
        previous = self.arcball(previous)
        current = self.arcball(current)
        angle = -np.arccos(min(1.0, np.asscalar(previous @ current)))
        if abs(angle) < 1e-6:
            return
        iV = np.linalg.inv(self.viewer.view_matrix[:3, :3])
        axis = iV @ np.cross(previous, current)
        axis = axis / np.linalg.norm(axis)
        a = np.cos(angle / 2.0)
        b, c, d = -axis * np.sin(angle / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        R = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                      [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                      [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
        self.cam_vec = R @ self.cam_vec
        self.up_vec = R @ self.up_vec
        self.update_view()

    def zoom(self, delta):
        self.zoom_level = self.zoom_level + delta
        self.update_view()
        self.update_perspective()

    def load_pointcloud(self, filename_or_pointcloud):
        if isinstance(filename_or_pointcloud, str):
            PC = pointcloud().readPLY(filename_or_pointcloud)
        else:
            PC = filename_or_pointcloud
        PC.colors = PC.colors / 255
        valid = ~(np.isnan(PC.positions).any(axis=1))
        valid &= ~(np.isnan(PC.normals).any(axis=1))
        valid &= ~(np.isnan(PC.colors).any(axis=1))
        PC.positions = PC.positions[valid]
        PC.normals = PC.normals[valid]
        PC.colors = PC.colors[valid]

        self.viewer.set_vertices(np.hstack((PC.positions,
                                            PC.normals,
                                            PC.colors)))
        self.center = PC.positions.mean(axis=0)
        norm = np.linalg.norm(self.center)
        self.zoom_scale = 300
        self.zoom_level = np.log(norm) * self.zoom_scale
        self.cam_vec = -(self.center / norm)
        self.up_vec = np.array([0., -1., 0.])
        self.update_view()

    def run(self):
        self.viewer.show()
        return_val = self.__app.exec_()
        return return_val


if __name__ == "__main__":
    _app = app()
    _app.run()
