

class News():
    """Corressponding mongodb document

    Attributes:
        title:
        content:
        topic:
        date:
        images:

    """

    #     _id = ''
    title = ''
    content = ''
    topic = ''
    date = ''
    images = []

    def show(self):
        print("Title: %s" % self.title)
        print("\tContent: %s" % self.content)
        print("\tTopic: %s" % self.topic)
        print("\tDate: %s" % self.date)

#     def __init__(self, title, topic,)


