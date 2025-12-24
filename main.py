from VectorKBManager import VectorKBManager

kb = VectorKBManager()
kb.add_document("test_doc.txt")
print(kb.get_overview())
print(kb.search("什么是制约GPU性能的关键因素"))
