import tkinter
import tkinter.messagebox
import os
window=tkinter.Tk()
window.title("Change Dectecting Tool")
window.geometry("400x340+100+100")
window.resizable(False, False)

count=0

def button1UP():
    os.chdir("untitled1")
    ret= os.system("venv1\\scripts\\python gdltest.py")
    if ret==0:
        tkinter.messagebox.showinfo("INFO","Success!")
    else:
        tkinter.messagebox.showinfo("INFO","Failed!")
    os.chdir("..")
def button2UP():
    os.chdir("untitled3")
    ret = os.system("venv\\scripts\\python main.py")
    if ret == 0:
        tkinter.messagebox.showinfo("INFO", "Success!")
    else:
        tkinter.messagebox.showinfo("INFO", "Failed!")
    os.chdir("..")
def button3UP():
    os.chdir("untitled2")
    ret = os.system("venv\\scripts\\python main.py")
    if ret == 0:
        tkinter.messagebox.showinfo("INFO", "Success!")
    else:
        tkinter.messagebox.showinfo("INFO", "Failed!")
    os.chdir("..")
def button4UP():
    os.chdir("C:\\Users\\DEWH\\untitled6")
    ret = os.system("venv\\scripts\\python STANR.py")
    if ret == 0:
        tkinter.messagebox.showinfo("INFO", "Success!")
    else:
        tkinter.messagebox.showinfo("INFO", "Failed!")
    os.chdir("..")
button1 = tkinter.Button(window, overrelief="solid", width=0, command=button1UP, repeatdelay=1000, repeatinterval=100, text="RCVA method",height=5)
button1.pack(side='top',fill="x")
button2 = tkinter.Button(window, overrelief="solid", width=0, command=button2UP, repeatdelay=1000, repeatinterval=100, text="Data Fusion method",height=5)
button2.pack(side='top',fill="x")
button3 = tkinter.Button(window, overrelief="solid", width=0, command=button3UP, repeatdelay=1000, repeatinterval=100, text="Sorted histogram method",height=5)
button3.pack(side='top',fill="x")
button4 = tkinter.Button(window, overrelief="solid", width=0, command=button4UP, repeatdelay=1000, repeatinterval=100, text="STANR",height=5)
button4.pack(side='top',fill="x")
window.mainloop()
