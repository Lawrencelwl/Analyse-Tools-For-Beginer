from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt

# Subclass QPushButton to create a custom button with hover effects
class HoverButton(QtWidgets.QPushButton):
    def __init__(self, parent=None):
        super(HoverButton, self).__init__(parent)
        self.default_background_color = "#cd67cd"
        self.hover_background_color = "#d9abff"
        self.clicked_background_color = "#83A4FF"
        self.setStyle(self.default_background_color)
        self.setCursor(Qt.PointingHandCursor)

    def setStyle(self, background_color):
        self.setStyleSheet(f"background-color: {background_color};\n"
                           "border-style:outset;\n"
                           "border-width:2px;\n"
                           "border-radius:15px;\n"
                           "border-color:white;")
        
    def setClickedStyle(self):
        self.setStyle(self.clicked_background_color)

    def setDefaultStyle(self):
        self.setStyle(self.default_background_color)

    # Override enterEvent
    def enterEvent(self, event):
        if self.styleSheet().split(";")[0] != f"background-color: {self.clicked_background_color}":
            self.setStyle(self.hover_background_color)
        super(HoverButton, self).enterEvent(event)

    # Override leaveEvent
    def leaveEvent(self, event):
        if self.styleSheet().split(";")[0] != f"background-color: {self.clicked_background_color}":
            self.setStyle(self.default_background_color)
        super(HoverButton, self).leaveEvent(event)