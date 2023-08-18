from tkinter import *
import torch
import torchvision.models as models
import PIL.ImageGrab as ImageGrab
from PIL import Image, ImageOps
import numpy as np
from NeuralNetTrainer import NeuralNet

#Drawing Application based on Jay Polra's model from Copy Assignment

class Write():
    def __init__(self, window):
        #Loads the NeuralNet for later use
        self.model = NeuralNet()
        self.model = torch.load("model.pth")
        
        #Drawing application
        self.window = window
        self.window.title("Letter and Number Identifier")
        self.window.geometry("560x560")
        self.window.configure(background="grey")

        self.write = "white"
        self.erase = "black"
        self.pointer = self.write

        self.eraserBtn = Button(self.window, text="Erase", bd=4, bg='grey', command = self.eraser, width=9, relief=RIDGE)
        self.eraserBtn.place(x=0, y=0)

        self.clearBtn = Button(self.window, text="Clear", bd=4, bg='grey', command = lambda : self.background.delete("all"), width=9, relief=RIDGE)
        self.clearBtn.place(x=90, y=0)

        self.analyzeBtn = Button(self.window, text="Analyze", bd=4, bg='grey', command = self.analyze, width=9, relief=RIDGE)
        self.analyzeBtn.place(x=180, y=0)

        self.background = Canvas(self.window,bg='black', bd=0, height=280, width=280)
        self.background.place(relx=0.5, rely=0.5, anchor=CENTER)


        self.background.bind("<B1-Motion>", self.draw)
        
        self.label = Label()

    def draw(self, event):
        x1,y1 = (event.x-2), (event.y-2)  
        x2,y2 = (event.x+2), (event.y+2)  

        self.background.create_oval(x1, y1, x2, y2, fill=self.pointer, outline=self.pointer, width="21")

    def eraser(self):
        if self.pointer != self.erase:
            self.pointer = self.erase
        else:
            self.pointer = self.write

    def analyze(self):
        #resets the label
        self.label.destroy()
        
        self.savePhoto()
        
        #Label list
        labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                  'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                  'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                  'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd',
                  'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                  'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
                  'y', 'z']
        
        #Turns the image into a usable tensor
        image = Image.open("Recent_Analysis.jpg").convert('L')
        image_vals = np.array(image).astype(np.float32)
        image_vals /= 255
        image_vals = torch.tensor(image_vals).unsqueeze(axis=0).unsqueeze(axis=0)
        
        #inputs the image into the Neural network
        with torch.no_grad():
            output = self.model(image_vals)
            
        outputIndex = torch.argmax(output)
            
        self.label = Label(self.window, text=f"Is your character {labels[outputIndex]}?", bd=4, bg='grey', relief=RIDGE, font=("Helvetica", 18))
        self.label.place(x=0, y=90)
        

    def savePhoto(self):
        try:
            recentPhoto = "Recent_Analysis.jpg"

            #Crops the Canvas
            x = self.window.winfo_rootx() + self.background.winfo_x() + 5
            y = self.window.winfo_rooty() + self.background.winfo_y() + 5

            x1 = x + self.background.winfo_width() - 10
            y1 = y + self.background.winfo_height() - 10

            image = ImageGrab.grab().crop((x, y, x1, y1))
            
            #Resizes and orients the photo so it can be fed to the Neural Net
            image.thumbnail((28, 28))
            image = ImageOps.flip(image)
            image = image.rotate(270)
            image.save(recentPhoto)

        except:
            print("Error in saving the photo for analysis")


if __name__ == "__main__":

    window = Tk()
    write = Write(window)
    window.mainloop()