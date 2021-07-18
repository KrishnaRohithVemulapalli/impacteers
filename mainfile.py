from tkinter import*
from PIL import ImageTk,Image
import tkinter.font as F
from sahKnn import *
from desciscratch import *
from lda import *
from SVC import *
from linearregression import *
from naivebayes import *

root = Tk()
root.title('Iris_flower_Classification')
root.iconbitmap('pics\\twoicon.ico')
# .geometry("window width x window height + position right + position down")
root.geometry('1360x680+-5+0')
root.configure(bg="#ffe3e3")
root.resizable(0, 0)
frame1=Frame(root,highlightbackground = "red", highlightcolor= "red",highlightthickness=3, width=900, height=600,bg="#e81a1a") ##e81a1a
frame1.grid(padx=10,pady=(20,0),row=0,column=0,sticky="nsew")

frame101=Frame(frame1,highlightbackground = "red", highlightcolor= "red",highlightthickness=3, width=860, height=500)
frame101.grid(row=0,column=0,padx=20,pady=10)

frame102=Frame(frame1,highlightbackground = "red", highlightcolor= "red",highlightthickness=3, width=860, height=50)
frame102.grid(row=1,column=0,padx=20,pady=5)

frame2=Frame(root,highlightbackground = "blue", highlightcolor= "blue",highlightthickness=3, width=400, height=600,bg="#2f39ed")
frame2.grid(padx=20,pady=(20,0),row=0,column=1)


image=Image.open("pics\\Slide1.JPG")
image = image.resize((875, 485), Image.ANTIALIAS)
myimg=ImageTk.PhotoImage(image)


image=Image.open("pics\\Slide2.JPG")
image = image.resize((875, 485), Image.ANTIALIAS)
myimg1=ImageTk.PhotoImage(image)

image=Image.open("pics\\Slide3.JPG")
image = image.resize((875, 485), Image.ANTIALIAS)
myimg2=ImageTk.PhotoImage(image)

image=Image.open("pics\\Slide4.JPG")
image = image.resize((875, 485), Image.ANTIALIAS)
myimg3=ImageTk.PhotoImage(image)

image=Image.open("pics\\Slide5.JPG")
image = image.resize((875, 485), Image.ANTIALIAS)
myimg4=ImageTk.PhotoImage(image)

image=Image.open("pics\\Slide6.JPG")
image = image.resize((875, 485), Image.ANTIALIAS)
myimg5=ImageTk.PhotoImage(image)

image=Image.open("pics\\Slide7.JPG")
image = image.resize((875, 485), Image.ANTIALIAS)
myimg6=ImageTk.PhotoImage(image)

image=Image.open("pics\\Slide8.JPG")
image = image.resize((875, 485), Image.ANTIALIAS)
myimg7=ImageTk.PhotoImage(image)

image=Image.open("pics\\Slide9.JPG")
image = image.resize((875, 485), Image.ANTIALIAS)
myimg8=ImageTk.PhotoImage(image)

image=Image.open("pics\\Slide10.JPG")
image = image.resize((875, 485), Image.ANTIALIAS)
myimg9=ImageTk.PhotoImage(image)

image=Image.open("pics\\Slide11.JPG")
image = image.resize((875, 485), Image.ANTIALIAS)
myimg10=ImageTk.PhotoImage(image)

image=Image.open("pics\\Slide12.JPG")
image = image.resize((875, 485), Image.ANTIALIAS)
myimg11=ImageTk.PhotoImage(image)

image=Image.open("pics\\Slide13.JPG")
image = image.resize((875, 485), Image.ANTIALIAS)
myimg12=ImageTk.PhotoImage(image)

image=Image.open("pics\\Slide14.JPG")
image = image.resize((875, 485), Image.ANTIALIAS)
myimg13=ImageTk.PhotoImage(image)










imagex1 = Image.open("pics\\Slide15.jpg")
imagex1 = imagex1.resize((875, 425), Image.ANTIALIAS)
seto = ImageTk.PhotoImage(imagex1)

imagex1 = Image.open("pics\\Slide16.jpg")
imagex1 = imagex1.resize((875, 425), Image.ANTIALIAS)
versi = ImageTk.PhotoImage(imagex1)

imagex1 = Image.open("pics\\Slide17.jpg")
imagex1 = imagex1.resize((875, 425), Image.ANTIALIAS)
virgi = ImageTk.PhotoImage(imagex1)









imglis=[myimg,myimg1,myimg2,myimg3,myimg4,myimg5,myimg6,myimg7,myimg8,myimg9,myimg10,myimg11,myimg12,myimg13]
global label
label=Label(frame101,text="Hello",image=myimg)
label.grid(row=0,column=0)
global i
i=0
def moveback():
    glis=globals()
    glis['i']=(i - 1)%14
    global label
    label.grid_forget()
    label = Label(frame101,text="Hello", image=imglis[i])
    label.grid(row=0, column=0, columnspan=3)


def movefront():
    glis=globals()
    glis['i']=(i + 1)%14
    global label
    label.grid_forget()
    label = Label(frame101,text="Hello", image=imglis[i])
    label.grid(row=0, column=0, columnspan=3)



font1=F.Font(family='Helvetica', size=25, weight='bold')
global button_back
button_back=Button(frame102,text="PREV",command=moveback)
# button_quit=Button(text="EXIT_PROGRAM",command=root.quit)
global button_for
button_for=Button(frame102,text="NEXT",command=movefront)
button_back['font']=font1
button_for['font']=font1


button_back.grid(row=1,column=0,padx=(0,5))
# button_quit.grid(row=1,column=1)
button_for.grid(row=1,column=2)

label1=Label(frame2,text="ALGORITHMS")
label1['font']=font1
label1.grid(row=0,column=0,padx=70,pady=20)




def takeinput(alg):
    for widgets in frame101.winfo_children():
        widgets.destroy()

    inlabel=Label(frame101,text="Enter The Following Dimensions in Cm's ")
    inlabel['font']=font1
    inlabel.grid(row=0,column=0)
    inlabel1=Label(frame101,text="Enter The Sepal Length : ",font=font1)
    inlabel1.grid(row=1,column=0,padx=(0,0),pady=20)
    inlabel2=Label(frame101,text="Enter The Sepal Width :   ",font=font1)
    inlabel2.grid(row=2,column=0,padx=(0,0),pady=20)
    inlabel3=Label(frame101,text="Enter The Petal Length :  ",font=font1)
    inlabel3.grid(row=3,column=0,padx=(0,0),pady=20)
    inlabel4=Label(frame101,text="Enter The Petal Width :    ",font=font1)
    inlabel4.grid(row=4,column=0,padx=(0,0),pady=(20))
    inlabel5 = Label(frame101, text="Solving Via "+alg, font=font1)
    inlabel5.grid(row=5, column=0, padx=(0, 0), pady=(30,72))
    global input1
    global input2
    global input3
    global input4
    input1=Entry(frame101,font=font1)
    input1.grid(row=1,column=1,padx=(0,20))
    input2=Entry(frame101,font=font1)
    input2.grid(row=2,column=1,padx=(0,20))
    input3=Entry(frame101,font=font1)
    input3.grid(row=3,column=1,padx=(0,20))
    input4=Entry(frame101,font=font1)
    input4.grid(row=4,column=1,padx=(0,20),pady=(20))
    font2 = F.Font(family='Helvetica', size=25, weight='bold')


    global button_home
    global button_enter
    for widgets in frame102.winfo_children():
        widgets.destroy()
    button_home = Button(frame102, text="HOME", command=gohome,font=font2)
    button_home.grid(row=1,column=0,padx=(0,5))


    button_enter = Button(frame102, text="ANSWER", command=lambda :answer(alg), font=font2)
    button_enter.grid(row=1, column=1)

def gohome():
    font2 = F.Font(family='Helvetica', size=25, weight='bold')

    for widgets in frame101.winfo_children():
        widgets.destroy()
    for widgets in frame102.winfo_children():
        widgets.destroy()
    global label
    label = Label(frame101, text="Hello", image=myimg)
    label.grid(row=0, column=0)
    global button_back
    global i
    i=0
    button_back = Button(frame102, text="PREV", command=moveback, font=font2)
    button_back.grid(row=1, column=0, padx=(0,5))
    global button_for
    button_for = Button(frame102, text="NEXT", command=movefront, font=font2)
    button_for.grid(row=1, column=1)

def resultframe(finalop):
    for widgets in frame101.winfo_children():
        widgets.destroy()
    for widgets in frame102.winfo_children():
        widgets.destroy()

    global button_home
    button_home = Button(frame102, text="HOME", command=gohome, font=font1)
    button_home.grid(row=1, column=0)
    label1=Label(frame101,text="THE MODEL TRAINED WITH ACCURACY "+str(finalop[0]*100.0)[:6],font=font1)
    label2=Label(frame101,text="FLOWER IS "+finalop[1],font=font1)
    label1.grid(row=0,column=0,padx=40,pady=0)
    if(finalop[1]=="Iris-setosa"):
        setosalabel = Label(frame101, text="hello", image=seto)
        setosalabel.grid(row=1,column=0)
    elif(finalop[1]=="Iris-versicolor"):
        versilabel = Label(frame101, text="hello", image=versi)
        versilabel.grid(row=1, column=0)
    elif(finalop[1]=="Iris-virginica"):
        virgilabel = Label(frame101, text="hello", image=virgi)
        virgilabel.grid(row=1, column=0)
    label2.grid(row=2,column=0,padx=40,pady=0)





def answer(alg):
    x1=input1.get()
    x2=input2.get()
    x3=input3.get()
    x4=input4.get()

    input1.delete(0,END)
    input2.delete(0,END)
    input3.delete(0,END)
    input4.delete(0,END)

    if(alg=="Decision Tree Algorithm"):
        finalop=solve_des(float(x1),float(x2),float(x3),float(x4))
        resultframe(finalop)
    elif(alg=="K-Nearest Neighbors"):
        finalop = solve_knn(float(x1), float(x2), float(x3), float(x4))
        resultframe(finalop)
    elif (alg == "LDA"):
        finalop = solve_lda(float(x1), float(x2), float(x3), float(x4))
        resultframe(finalop)
    elif (alg == "Linear regression"):
        finalop = solve_linearregression(float(x1), float(x2), float(x3), float(x4))
        resultframe(finalop)
    elif (alg == "SVC"):
        finalop = solve_SVC(float(x1), float(x2), float(x3), float(x4))
        resultframe(finalop)
    elif (alg == "Naive Bayes"):
        finalop = solve_bynb(float(x1), float(x2), float(x3), float(x4))
        resultframe(finalop)


font1=F.Font(family='Helvetica', size=20, weight='bold')

Button1=Button(frame2,text="k-Nearest Neighbors",width=20,command=lambda:takeinput('K-Nearest Neighbors'))
Button1['font']=font1
Button1.grid(row=1,column=0,pady=10)


Button2=Button(frame2,text="Linear regression",width=20,command=lambda:takeinput('Linear regression'))
Button2['font']=font1
Button2.grid(row=2,column=0,pady=10)


Button3=Button(frame2,text="LDA",width=20,command=lambda :takeinput('LDA'))
Button3['font']=font1
Button3.grid(row=3,column=0,padx=0,pady=10)


Button4=Button(frame2,text="Decision Tree",width=20,command=lambda :takeinput('Decision Tree Algorithm'))
Button4['font']=font1
Button4.grid(row=4,column=0,padx=0,pady=10)


Button5=Button(frame2,text="Naive Bayes",width=20,command=lambda :takeinput('Naive Bayes'))
Button5['font']=font1
Button5.grid(row=5,column=0,padx=0,pady=10)

Button6=Button(frame2,text="Support Vector Classifier",width=20,command=lambda:takeinput('SVC'))
Button6['font']=font1
Button6.grid(row=6,column=0,padx=0,pady=10)






root.mainloop()