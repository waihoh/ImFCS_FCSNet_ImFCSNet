def appendtofile(filename, text):
    f = open(filename, "a+")
    f.write(text)
    f.close()