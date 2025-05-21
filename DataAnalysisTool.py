#data analysis tool NEW


#IMPORTS
import os                  #File handling
import pandas as pd      #data handling
import matplotlib.pyplot as plt #Plotting
from matplotlib import animation, colors
import numpy as np #math functions
import math
from sklearn.model_selection import KFold, cross_val_score #kfold
import xlwt  #handling excel files
import re   #input verification

#import files
import DAExtraFunction as ef


#tkinter user interface
from tkinter import *    
from tkinter import ttk
from tkinter import messagebox
from tkinter import simpledialog



#Main Window Class
class MainWindow:
    def __init__(self, master):
        self.master = master
        self.dataframe = []
        
        #Set window title and size
        self.master.title("Data Analysis Tool")        
        self.master.geometry("600x550")


        #Import external data
        self.btnImp = Button(self.master, text = "Import Data", command=self.importData)
        self.btnImp.place(x=10, y=20)

        #View Data
        self.btnView = Button(self.master, text = "View Data", command = self.ViewData, state=DISABLED)
        self.btnView.place(x=10, y=60)

        #Edit Data
        self.btnEdit = Button(self.master, text = "Edit Data", command = self.EditData, state=DISABLED)
        self.btnEdit.place(x=100, y=60)

        #Export data
        self.btnExport = Button(self.master, text = "Export Data", command = self.Export, state=DISABLED)
        self.btnExport.place(x=10, y=100)

        self.lblExp = Label(self.master, text="File Type:")
        self.lblExp.place(x=200, y=80)

        self.cmbVarExp = StringVar()
        self.ExpValues = ["csv", "xls"]
        self.cmbExp = ttk.Combobox(self.master, textvariable=self.cmbVarExp, state="readonly", values=self.ExpValues, width=5)
        self.cmbExp.place(x=200, y=100)
        self.cmbExp.set(self.ExpValues[0])

        #Principle Component Analysis
        self.lblPCA = Label(self.master, text="Dimensions:")
        self.lblPCA.place(x=200, y=130)
        
        self.cmbVarPCA = StringVar()
        self.PCAValues = [2,3]
        self.cmbPCA = ttk.Combobox(self.master, textvariable=self.cmbVarPCA, state="readonly", values=self.PCAValues, width=1)
        self.cmbPCA.place(x=200, y=150)
        self.cmbPCA.set(self.PCAValues[0])
        
        self.btnPCA = Button(self.master, text = "Principle Component Analysis", command = self.PCA, state= DISABLED)
        self.btnPCA.place(x=10, y=150)

        #K means clustering
        self.btnKMeans = Button(self.master, text = "K-means Clustering", command= self.KMeans, state= DISABLED)
        self.btnKMeans.place(x=10, y=220)

        self.lblCluster = Label(self.master, text="Clusters:")
        self.lblCluster.place(x=200, y=200)

        validation = self.master.register(self.validateNumeric)
        self.entClusters = Entry(self.master, validate="key", validatecommand=(validation, "%P"), width=4)
        self.entClusters.place(x=200, y=220)

        self.lblIter = Label(self.master, text="Iterations:")
        self.lblIter.place(x=280, y=200)

        self.entIter = Entry(self.master, validate="key", validatecommand=(validation, "%P"), width=6)
        self.entIter.place(x=280, y=220)
                
        #Parallel Coordinates Plot
        self.btnPCP = Button(self.master, text = "Parallel Coordinates Plot", command=self.PCP, state= DISABLED)
        self.btnPCP.place(x=10, y=270)

        #Box plot
        self.btnBox = Button(self.master, text = "Box plot", command=self.Box, state= DISABLED)
        self.btnBox.place(x=10, y=320)

        dataTop = ["Empty"]
        self.cmbVarHead = StringVar()
        self.cmbHead = ttk.Combobox(self.master, textvariable=self.cmbVarHead, state="readonly", values= dataTop)
        self.cmbHead.place(x= 200, y=320)

        #Scatter plot
        self.btnScatter = Button(self.master, text = "Scatter Plot", command= self.Scatter, state=DISABLED)
        self.btnScatter.place(x=10, y=350)

        self.cmbVarHead2 = StringVar()
        self.cmbHead2 = ttk.Combobox(self.master, textvariable=self.cmbVarHead2, state="readonly", values= dataTop)
        self.cmbHead2.place(x= 200, y=350)

        #KFold cross-validation
        self.btnKFold = Button(self.master, text = "K-fold cross-validation", command= self.KFold, state=DISABLED)
        self.btnKFold.place(x=10, y=400)

        self.cmbVarTarget = StringVar()
        self.cmbTarget = ttk.Combobox(self.master, textvariable=self.cmbVarTarget, state="readonly", values= dataTop)
        self.cmbTarget.place(x= 200, y=400)

        self.entFold = Entry(self.master, validate="key", validatecommand=(validation, "%P"), width=4)
        self.entFold.place(x=400, y=400)

        self.lblTarget = Label(self.master, text="Target:")
        self.lblTarget.place(x=200, y=379)

        self.lblFold = Label(self.master, text="Folds:")
        self.lblFold.place(x=400, y=379)

        #KFold Prediction
        self.btnKPred = Button(self.master, text = "K-fold cross-validation prediction", command= self.KPred, state=DISABLED)
        self.btnKPred.place(x=10, y=425)

        #Self organising maps
        self.btnSOM = Button(self.master, text = "Self organising map", command= self.SOM, state=DISABLED)
        self.btnSOM.place(x=10, y=475)

        self.lblSize = Label(self.master, text="Size:")
        self.lblSize.place(x=150, y=454)

        self.entSize = Entry(self.master, validate="key", validatecommand=(validation, "%P"), width=4)
        self.entSize.place(x=150, y=475)

        self.lblSigma = Label(self.master, text="Max M Distance:")
        self.lblSigma.place(x=200, y=454)

        self.entSigma = Entry(self.master, validate="key", validatecommand=(validation, "%P"), width=4)
        self.entSigma.place(x=200, y=475)

        self.lblLrate = Label(self.master, text="Learning Rate:")
        self.lblLrate.place(x=350, y=454)

        self.entLrate = Entry(self.master, validate="key", width=4)
        self.entLrate.place(x=350, y=475)

        self.lblEpoch = Label(self.master, text="Epochs:")
        self.lblEpoch.place(x=450, y=454)

        self.entEpoch = Entry(self.master, validate="key", validatecommand=(validation, "%P"), width=4)
        self.entEpoch.place(x=450, y=475)



        
    #Importing data
    #Open Import data window
    def importData(self):
        print("Import Data Window Created")
        self.master.withdraw() #hide main window
        impDataWin = WinImpData(self.master, self.winClose, self.returnData)


    #Return imported data to main window
    def returnData(self, data):
        print("Returning data to Main window")
        self.dataframe = data
        #convert dataframe to array
        #filter out string column
        numeric_columns = self.dataframe.select_dtypes(include=[np.number]).columns
        filtered_df = self.dataframe[numeric_columns]
        self.data = filtered_df.values
        self.cluster = np.zeros(len(self.data))
        self.dataTrans = []
        #show main window again
        self.master.deiconify()

        #enable all functionality
        self.btnView.configure(state=NORMAL)
        self.btnEdit.configure(state=NORMAL)
        self.btnExport.configure(state=NORMAL)
        self.btnPCA.configure(state=NORMAL)
        self.btnPCP.configure(state=NORMAL)
        self.btnBox.configure(state=NORMAL)
        self.btnScatter.configure(state=NORMAL)
        self.btnKMeans.configure(state=DISABLED)
        self.btnKFold.configure(state=NORMAL)
        self.btnKPred.configure(state=DISABLED)

        if len(numeric_columns) == self.dataframe.shape[1]:
            self.btnSOM.configure(state=NORMAL)
        
        self.dataHead = self.dataframe.columns.tolist()
        self.dataTop=[]
        for i in numeric_columns:
            self.dataTop.append(i)
        self.cmbHead['values'] = self.dataTop
        self.cmbHead2['values'] = self.dataTop
        self.cmbTarget['values'] = self.dataHead
        self.cmbHead.set(self.dataTop[0])
        self.cmbHead2.set(self.dataTop[0])
        self.cmbTarget.set(self.dataHead[-1])



    #return to main window without change
    def winClose(self):
        print("Returning to main window")
        self.master.deiconify()

    #View data
    def ViewData(self):
        self.master.withdraw()
        ViewDataWin = WinViewData(self.master, self.dataframe, self.winClose)

    #Edit data
    def EditData(self):
        self.master.withdraw()
        EditDataWin = WinEditData(self.master, self.dataframe, self.winEditClose)

    #Exit function when edit data window closes
    def winEditClose(self, df):
        print("Returning to main window")
        self.dataframe = df
        
        #update combo boxes
        numeric_columns = self.dataframe.select_dtypes(include=[np.number]).columns
        self.dataHead = self.dataframe.columns.tolist()
        self.dataTop=[]
        for i in numeric_columns:
            self.dataTop.append(i)
        self.cmbHead['values'] = self.dataTop
        self.cmbHead2['values'] = self.dataTop
        self.cmbTarget['values'] = self.dataHead
        self.cmbHead.set(self.dataTop[0])
        self.cmbHead2.set(self.dataTop[0])
        self.cmbTarget.set(self.dataHead[-1])

        if len(numeric_columns) == self.dataframe.shape[1]:
            self.btnSOM.configure(state=NORMAL)
        
        self.master.deiconify()

    #Export data as csv or xls
    def Export(self):
        fileType = self.cmbVarExp.get()
        fileName = simpledialog.askstring("Export data", "Enter the name of the file (without .csv/.xls)")
        file = ""
        if fileType == "csv":
            file = fileName + ".csv"
            self.dataframe.to_csv(file, index=False)
        else:
            file = fileName + ".xls"
            self.dataframe.to_excel(file, index=False, engine='xlwt')
        mess = "File exported as :" + file
        messagebox.showinfo("Export data", mess)
        
    #Validate numeric input
    def validateNumeric(self, newVal):
        if newVal.isdigit() or newVal == "":
            return True
        else:
            return False

    #Alert message if no excel sheet exists
    def alrt(self, message):
        messagebox.showinfo("Error", message)

    #Perform Principle Component Analysis
    def PCA(self):
        #Get the desired dimensions
        Dim = int(self.cmbVarPCA.get())

        #get the new data set
        self.dataTrans = ef.PCA(self.data, Dim)
        #print(self.dataTrans)
        #plot the data
        fig = ef.PCAPlot(self.dataTrans, Dim)
        fig.show()
        #enable k means clustering
        self.btnKMeans.configure(state=NORMAL)

    #K means clustering
    def KMeans(self):
        cluster = self.entClusters.get()
        iterations = self.entIter.get()

        if cluster == "" or iterations == "": #if missing fields
            self.alrt("Error, missing information needed")
        else:
            self.cluster = ef.kmeans(self.dataTrans, int(cluster), int(iterations)) #perform k means clustering
            fig = ef.kmeansPlot(self.cluster, self.dataTrans) #plot clusters
            fig.show()

    #Parallel Coordinates Plot   
    def PCP(self):
        fig = ef.PCP(self.data, self.cluster, self.dataTop)
        fig.show()

    #Box and whisker plot
    def Box(self):
        fig = plt.figure(figsize =(10, 7))
        dataValue = self.dataframe[self.cmbVarHead.get()].tolist()
        plt.boxplot(dataValue)
        plt.ylabel(self.cmbVarHead.get())
        plt.show()

    #Scatter plot
    def Scatter(self):
        fig = plt.figure(figsize =(10, 7))
        dataValue = self.dataframe[self.cmbVarHead.get()].tolist()
        dataValue2 = self.dataframe[self.cmbVarHead2.get()].tolist()
        plt.scatter(dataValue, dataValue2)
        plt.xlabel(self.cmbVarHead.get())
        plt.ylabel(self.cmbVarHead2.get())
        plt.show()
        #enable kmeans clustering
        self.btnKMeans.configure(state=NORMAL)

        #Get new transformed data
        self.dataTrans = self.dataframe[[self.cmbVarHead.get(), self.cmbVarHead2.get()]].values

    #Kfold cross-validation
    def KFold(self):
        if self.entFold.get() == "":
            self.alrt("Error, missing information needed")
            return
        KTarget = self.cmbTarget.get()
        target = self.dataframe[KTarget].values
        #print(target)
        self.otherCol = [col for col in self.dataHead if col != KTarget]
        kData = self.dataframe[self.otherCol].values
        #print(kData)
        self.model, score = ef.kfold(kData, target, int(self.entFold.get()))
        #print(score)
        mess = "Accuracy of model is: " + str(100*np.mean(score)) + "%"
        messagebox.showinfo("KFold Cross-validation", mess)
        self.btnKPred.configure(state=NORMAL)

    #Predictive K fold cross-validation
    #Predict target value given data, using K Fold model obtained
    def KPred(self):
        #testData = [5.1, 3.5, 1.4, 0.2]
        testData = []
        for i in self.otherCol:
            mess = "Enter " + i + ": "
            dataPoint = simpledialog.askstring("Input Data", mess)
            testData.append(float(dataPoint))
        #print(testData)
        testData = np.array(testData).reshape(1, -1)
        prediction = self.model.predict(testData)
        mess = "The model predicts: " + str(prediction)
        messagebox.showinfo("KFold Cross-validation", mess)

    #self organising data
    def SOM(self):
        kData = np.array(self.dataframe[self.dataTop])
        size = self.entSize.get()
        if size == "":
            size = int(math.sqrt(5*math.sqrt(len(kData))))
        else:
            size = int(size)
        fig = ef.SOM(kData, self.dataframe[self.cmbTarget.get()].values, size, int(self.entSigma.get()), float(self.entLrate.get()), int(self.entEpoch.get()))
        fig.show()

        



#Import data window
class WinImpData:
    def __init__(self, master, winClose, returnData):
        self.master = master
        self.top = Toplevel(master)
        #Protocol for close even
        self.top.protocol("WM_DELETE_WINDOW", self.on_close)
        #Set title and sizes
        self.top.title("Import Data")        
        self.top.geometry("300x100")
        self.frame = Frame(self.top)

        #External Functions
        self.returnData = returnData
        self.winClose = winClose

        #Import data button
        self.btnimp = Button(self.top, text = "Import", command = self.imp)
        self.btnimp.pack(pady = 10)

        #Combobox to select desired data file
        self.ExcelFiles = self.getFiles()
        self.cmbVar = StringVar()
        self.cmbFiles = ttk.Combobox(self.top, textvariable=self.cmbVar, state="readonly", values=self.ExcelFiles)
        self.cmbFiles.pack(pady=10)
        self.cmbFiles.set(self.ExcelFiles[0])

        self.frame.pack(padx=10, pady=10)

    #get all excel files
    def getFiles(self):
        currentDir = os.getcwd() # get current directory
        files = os.listdir(currentDir) #list all files in current directory
        file_list = [file for file in files if file.endswith('.xls') or file.endswith('.csv')] #filter out files
        #check if there exists an excel spreadsheet
        if len(file_list) == 0:
            self.alrtNoFile()
        return file_list

    #read data
    def readData(self):
        file = self.cmbVar.get()
        print("Opening ", file)
        if file.endswith('.xls'):
            dataframe = pd.read_excel(file)
            #dataframe = dataframe.loc[:, ~dataframe.row.str.contains('^Unnamed')]
            #dataframe = dataframe.dropna(how='all')
        else:
            dataframe = pd.read_csv(file)
        print(dataframe.head())
        return dataframe

    
    #Import data and send back to main window
    def imp(self):
        data = self.readData()
        self.returnData(data)
        self.top.destroy()

    #Alert message if no excel sheet exists
    def alrtNoFile(self):
        messagebox.showinfo("No files found", "No .xls or .csv file was found in the current directory")
        #self.importDataClose()
        #self.top.destroy()
        self.on_close()

    #Close window without changing anything and go back to main window
    def on_close(self):
        self.winClose()
        self.top.destroy()


#View data window
class WinViewData:
    def __init__(self, master, dataframe, winClose):
        self.master = master
        self.top = Toplevel(master)
        #Protocol for close even
        self.top.protocol("WM_DELETE_WINDOW", self.on_close)
        #Set title and sizes
        self.top.title("View Data")        
        self.top.geometry("1500x500")
        self.winClose = winClose
        self.dataframe = dataframe
        self.frame = Frame(self.top)

        #Treeview to view and edit data
        self.tvData = ttk.Treeview(self.top)
        self.tvData.pack(expand=True, fill="both")

        self.addData()

        self.frame.pack(padx=10, pady=10)

    #Add data to treeview
    def addData(self):
        #Clear existing data
        self.tvData.delete(*self.tvData.get_children())
        columns = self.dataframe.columns.tolist() #get column names
        self.tvData['columns'] = columns
        self.tvData.heading("#0", text="Index")

        for column in columns:
            self.tvData.heading(column, text=column)
            self.tvData.column(column, stretch=YES)

        #Load data
        for i, row in self.dataframe.iterrows():
            values = row.values.tolist()
            self.tvData.insert("", END, text=str(i), values=values)

    #Close window without changing anything and go back to main window
    def on_close(self):
        self.winClose()
        self.top.destroy()



#New window to edit data
class WinEditData:
    def __init__(self, master, dataframe, winEditClose):
        self.master = master
        self.top = Toplevel(master)
        #Protocol for close even
        self.top.protocol("WM_DELETE_WINDOW", self.on_close)
        #Set title and sizes
        self.top.title("Edit Data")        
        self.top.geometry("300x200")
        self.winEditClose = winEditClose
        self.dataframe = dataframe

        #add data
        self.btnAdd = Button(self.top, text = "Add Data", command=self.addData)
        self.btnAdd.place(x=10, y=20)

        #delete data
        self.btnDelete = Button(self.top, text = "Delete Data", command=self.deleteData)
        self.btnDelete.place(x=10, y=70)

        self.lblIndex = Label(self.top, text="Index:")
        self.lblIndex.place(x=150, y=49)

        indexVal = [str(i) for i in range(len(self.dataframe))]
        self.cmbVarIndex = StringVar()
        self.cmbIndex = ttk.Combobox(self.top, textvariable=self.cmbVarIndex, state="readonly", values= indexVal)
        self.cmbIndex.place(x= 150, y=70)
        self.cmbIndex.set(indexVal[0])

        #Detect duplicates
        self.btnDupl = Button(self.top, text = "Detect Dupliates", command=self.detectDuplicate)
        self.btnDupl.place(x=10, y=120)

        #Enumerate data
        self.btnEnum = Button(self.top, text = "Enumerate data", command=self.Enum)
        self.btnEnum.place(x=10, y=170)
        
        self.enumVal = []
        for column in self.dataframe.columns:
             if not self.dataframe[column].apply(lambda x: isinstance(x, (int, float))).any():
                 self.enumVal.append(column)
        self.cmbVarEnum = StringVar()
        self.cmbEnum = ttk.Combobox(self.top, textvariable=self.cmbVarEnum, state="readonly", values= self.enumVal)
        self.cmbEnum.place(x=150, y=170)
        self.cmbEnum.set(self.enumVal[0])
        
    #Add data to dataframe
    def addData(self):
        self.dataHead = self.dataframe.columns.tolist()
        newData = []
        for i in self.dataHead:
            mess = "Enter " + i + ": "
            dataPoint = simpledialog.askstring("Input Data", mess)
            if dataPoint.isdigit():
                dataPoint = int(dataPoint)
            try:
                dataPoint = float(dataPoint)
            except ValueError:
                temp =1
            newData.append(dataPoint)
        newRow = pd.Series(newData, index = self.dataframe.columns)
        print(newRow)
        self.dataframe = self.dataframe.append(newRow, ignore_index=True)
        indexVal = [str(i) for i in range(len(self.dataframe))]
        self.cmbIndex['values'] = indexVal

    #Delete data from dataframe based on index
    def deleteData(self):
        self.dataframe = self.dataframe.drop(int(self.cmbVarIndex.get())).reset_index(drop=True)
        mess = "Deleted row " + self.cmbVarIndex.get()
        messagebox.showinfo("Delete data", mess)
        indexVal = [str(i) for i in range(len(self.dataframe))]
        self.cmbIndex['values'] = indexVal


        
    #Detect duplicate data
    def detectDuplicate(self):
        dupl = self.dataframe[self.dataframe.duplicated()].index.tolist()
        mess = "The index for duplicates are: " + str(dupl)
        messagebox.showinfo("Duplicates", mess)

    #Enumerate string values
    def Enum(self):
        enumCol = self.cmbVarEnum.get()
        self.dataframe[enumCol] = self.dataframe[enumCol].apply(lambda x: list(self.dataframe[enumCol].unique()).index(x))
        mess = "Column " + str(enumCol) + " has been updated"
        messagebox.showinfo("Enumerate data", mess)
        self.enumVal = []
        for column in self.dataframe.columns:
             if not self.dataframe[column].apply(lambda x: isinstance(x, (int, float))).any():
                 self.enumVal.append(column)
        self.cmbEnum['values'] = self.enumVal
        self.cmbEnum.set(self.enumVal[0])

    #On window close
    def on_close(self):
        self.winEditClose(self.dataframe)
        self.top.destroy()
        


#main function
def main():
    print("Start")

    root = Tk()
    window = MainWindow(root)
    root.mainloop()

#START
main()

print("Exit")
