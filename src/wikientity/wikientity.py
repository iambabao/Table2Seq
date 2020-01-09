import json


class WikiEntity:
    def __init__(self, js, idkey='PAGEID', sentkey='Sentences'):
        self.box = js if isinstance(js, dict) else json.loads(js)
        self.id = self.box[idkey]
        self.desc = self.box[sentkey]
        self.box.pop(idkey)
        self.box.pop(sentkey)

    def get_desc(self, as_list=False, first=True):
        if as_list:
            return self.desc
        elif first:
            return self.desc[0]
        else:
            return ' '.join(self.desc)

    def get_box(self):
        return self.box

    def get_properties(self):
        return list(self.box.keys())

    def get_value(self, p):
        return self.box.get(p)
