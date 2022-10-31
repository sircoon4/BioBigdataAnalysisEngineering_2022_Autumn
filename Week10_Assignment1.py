# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 09:27:06 2022

@author: sirco
"""

class MyStatus:
    age = 0
    name = ""
    height = 0
    weight = 0
    
    def __init__(self, age, name, height, weight):
        self.age = age
        self.name = name
        self.height = height
        self.weight = weight
        
    def print_age(self):
        print("%s의 나이: %d" % (self.name, self.age))
    
    def print_name(self):
        print("이름: %s" % (self.name))
    
    def print_height(self):
        print("%s의 키: %d" % (self.name, self.height))
    
    def print_weight(self):
        print("%s의 몸무게: %d" % (self.name, self.weight))
    
status1, status2 = None, None

status1 = MyStatus(1, "이름1", 11, 111)
status2 = MyStatus(2, "이름2", 22, 222)

status1.print_name()
status1.print_age()
status1.print_height()
status1.print_weight()

status2.print_name()
status2.print_age()
status2.print_height()
status2.print_weight()