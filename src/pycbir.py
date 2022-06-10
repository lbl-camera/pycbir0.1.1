import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from interface.gui import Ui_pyCBIR

app = QApplication(sys.argv)
window = QMainWindow()
ui = Ui_pyCBIR()
ui.setupUi(window)
ui.groupBox.setVisible(True)
window.show()
sys.exit(app.exec_())
