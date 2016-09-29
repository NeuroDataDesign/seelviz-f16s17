from clarityviz import clarityviz

testtoken = 'testtoken'
c = clarityviz(testtoken)

if (c._token == testtoken):
    print('Token initialization success')
else:
    print('Token initialization failure')
