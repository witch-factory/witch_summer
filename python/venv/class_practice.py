class calculator:
    def __init__(self,first,second):
        self.first=first
        self.second=second

    def setdata(self,first,second):
        self.first=first
        self.second=second

    def add(self):
        return self.first+self.second

cal1=calculator(10,14)
print(cal1.add())
cal1.setdata(14,15)
print(cal1.add())