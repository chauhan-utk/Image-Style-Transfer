import sys

from PyQt5 import QtGui , QtCore,QtWidgets
from PyQt5.Qt import QDesktopWidget, QWidget
import scipy.misc, numpy as np
from transfer_style_fast import transfer_style

class Window(QtWidgets.QMainWindow):
    
    
    def __init__(self):
        super(Window, self).__init__()
        self.setWindowTitle("Image Style Transfer ")
        self.setWindowIcon(QtGui.QIcon('golden_gate.png'))
        #dw = QDesktopWidget()
        #self.setFixedSize(dw.width(),dw.height())
        openImageAction = QtWidgets.QAction("&Open Image", self)
        openImageAction.setShortcut("Ctrl+O")
        openImageAction.setStatusTip('Open Image')
        openImageAction.triggered.connect(self.image_open)

        saveImageAction = QtWidgets.QAction("&Save Image", self)
        saveImageAction.setShortcut("Ctrl+S")
        saveImageAction.setStatusTip('Save Image')
        saveImageAction.triggered.connect(self.image_save)

        quitAction = QtWidgets.QAction("&Exit",self)
        quitAction.setShortcut("Ctrl+Q")
        quitAction.setStatusTip('Exit')
        quitAction.triggered.connect(self.close_application)

        
        mainMenu= self.menuBar()
        imageMenu = mainMenu.addMenu('&Image')
        imageMenu.addAction(openImageAction)
        imageMenu.addAction(saveImageAction)
        imageMenu.addAction(quitAction)
        
        self.status = 0
        self.imageStyles = ['starry_night', 'starry_night_2', 'starry_night_diff', 'shipwreck_colors', 'wave_2000', 'wave_4000', 'wave', 'rain_princess' ]
        self.percentages = ["0% 100%", "10% 90%","20% 80%", "30% 70%", "40% 60%", "50% 50%", "60% 40%", "70% 30%", "80% 20%", "90% 10%", "100% 0%"]
        self.currentPercentageIndex = 0
        self.currentStyleIndex = 0
        self.curr_img = None
        self.home()
        
    def button_widget(self):
        self.buttonWidget = QtWidgets.QWidget()
        gridLayout = QtWidgets.QGridLayout()
        self.btn=QtWidgets.QPushButton("Transfer")
        self.btn.clicked.connect(self.transfer)
        gridLayout.addWidget(self.btn,0,0)
        
        self.progress=QtWidgets.QProgressBar()   
        gridLayout.addWidget(self.progress,0,1)

        self.comboBox_1 = QtWidgets.QComboBox()
        for i in self.imageStyles:
            self.comboBox_1.addItem(i)
        self.comboBox_1.currentTextChanged.connect(self.change_style)
        gridLayout.addWidget(self.comboBox_1,1,0)
        
        self.comboBox_2 = QtWidgets.QComboBox(self)
        for i in self.percentages:
            self.comboBox_2.addItem(i)
        self.comboBox_2.currentTextChanged.connect(self.change_percentage)
        gridLayout.addWidget(self.comboBox_2,1,1)
        self.buttonWidget.setLayout(gridLayout)


    def style_widget(self):
        self.styleWidget = QtWidgets.QWidget()
        HBoxLayout = QtWidgets.QVBoxLayout()
        label = QtWidgets.QLabel('Style Image')
        HBoxLayout.addWidget(label)
        self.label_1 = QtWidgets.QLabel()
        self.pixMap_1 = QtGui.QPixmap(self.imageStyles[self.currentStyleIndex])
        #self.pixMap_1.scaled(QtCore.QSize(50,50), QtCore.Qt.IgnoreAspectRatio);
        self.label_1.setPixmap(self.pixMap_1)
        HBoxLayout.addWidget(self.label_1)
        self.styleWidget.setLayout(HBoxLayout)
        
    def input_widget(self):
        
        self.inputWidget = QtWidgets.QWidget()
        HBoxLayout = QtWidgets.QVBoxLayout()
        label = QtWidgets.QLabel('Input Image')
        HBoxLayout.addWidget(label)
        self.label_2 = QtWidgets.QLabel()
        self.pixMap_2 = QtGui.QPixmap('blank.JPG')
        self.label_2.setPixmap(self.pixMap_2)
        HBoxLayout.addWidget(self.label_2)
        self.inputWidget.setLayout(HBoxLayout)

    def output_widget(self):
        self.outputWidget = QtWidgets.QWidget()
        HBoxLayout = QtWidgets.QVBoxLayout()
        label = QtWidgets.QLabel('Output Image')
        HBoxLayout.addWidget(label)
        self.label_3 = QtWidgets.QLabel()
        self.pixMap_3 = QtGui.QPixmap('blank.JPG')
        self.label_3.setPixmap(self.pixMap_3)
        HBoxLayout.addWidget(self.label_3)
        self.outputWidget.setLayout(HBoxLayout)
    


    def home(self):
        
        self.generalWidget = QtWidgets.QWidget()
        gridLayout = QtWidgets.QGridLayout()
        self.button_widget()       
        gridLayout.addWidget(self.buttonWidget,0,0)
        
        self.style_widget()
        gridLayout.addWidget(self.styleWidget,0,1)
        
        self.input_widget()
        gridLayout.addWidget(self.inputWidget,1,0)
        
        self.output_widget()
        gridLayout.addWidget(self.outputWidget,1,1)
        self.generalScrollArea = QtWidgets.QScrollArea()
        self.generalWidget.setLayout(gridLayout)
        self.generalScrollArea.setWidget(self.generalWidget)
        self.generalScrollArea.setWidgetResizable(True)
        self.setCentralWidget(self.generalScrollArea)
        
        self.show()
        

    def image_save(self):
        
        self.pixMap_3 = self.label_3.pixmap()
        fileName = QtWidgets.QFileDialog.getSaveFileName(self, "Save File","","Images (*.png *.xpm *.jpg)");
        if fileName[0]:
            self.pixMap_3.save(fileName[0],'PNG')	
        
    def image_open(self):
        fileName = QtWidgets.QFileDialog.getOpenFileName(self,'Open Image',"","Images (*.png *.xpm *.jpg)")
        if fileName[0]:
            self.pixMap_2.load(fileName[0])
            self.label_2.setPixmap(self.pixMap_2)
            self.curr_img = scipy.misc.imread(fileName[0]).astype(np.float)

    def change_style(self,text):
        self.currentStyleIndex = self.comboBox_1.currentIndex()
        self.pixMap_1.load('%s.jpg'%self.imageStyles[self.currentStyleIndex])
        self.pixMap_1.scaled(QtCore.QSize(20,20), QtCore.Qt.KeepAspectRatio);
        self.label_1.setPixmap(self.pixMap_1)
        
    
    def change_percentage(self):
        self.currentPercentageIndex = self.comboBox_2.currentIndex()
        try:
            self.pixMap_3.load('out/out_%s_%s.jpg'%(self.currentPercentageIndex*10,self.imageStyles[self.currentStyleIndex]))
            self.label_3.setPixmap(self.pixMap_3)
        except:
            pass
        
    def transfer(self):
        if self.curr_img == None:
            return
        
        
        self.completed=0
        
        
        resized_image = scipy.misc.imresize(self.curr_img, (500, 500, 3))
        stylized_image = transfer_style(resized_image, self.imageStyles[self.currentStyleIndex])
        stylized_image = scipy.misc.imresize(stylized_image, self.curr_img.shape)
        
        w = 0
        while w <= 100:
            scipy.misc.imsave('out/out_%s_%s.jpg'%(w,self.imageStyles[self.currentStyleIndex]),w/100*stylized_image + (100-w)/100*self.curr_img)
            w+= 10
            
        #unused progress bar right now
        while self.completed <100:
            self.completed +=0.0001
            self.progress.setValue(self.completed)
        
        self.comboBox_2.setCurrentIndex(5)
        self.pixMap_3.load('out/out_50_%s.jpg'%(self.imageStyles[self.currentStyleIndex]))
        self.label_3.setPixmap(self.pixMap_3)
        
        
    def close_application(self):
        if self.status == 0:
            choice = QtWidgets.QMessageBox.question(self, 'Close',"Are you sure do you want to exit without saving",QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        
            if choice == QtWidgets.QMessageBox.Yes:
                sys.exit()
            else:
                pass


def run():        

    app = QtWidgets.QApplication(sys.argv)
    GUI = Window()
    sys.exit(app.exec_())



if __name__ == "__main__":
    run()
