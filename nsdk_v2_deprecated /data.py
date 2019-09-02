from collections import OrderedDict

class AllData():
    def __init__(self):
        self.person_rect = OrderedDict()
        self.face_rect = OrderedDict()
        self.face_age = OrderedDict()
        self.face_gender = OrderedDict()

    def reg_person_rect(self, ID, rects):
        self.person_rect[ID] = rects
    
    def person_rect_query(self, ID):
        pass