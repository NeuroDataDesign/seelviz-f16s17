import boto3

s3 = boto3.resource('s3')

bucket = 'demo-alee156'
key = 'albert.txt'

client = boto3.client('s3')
client.delete_object(Bucket=bucket, Key='albert2.txt')

obj = s3.Object(bucket, key)
data2 = obj.get()['Body'].read().decode('utf-8')
print data2

data3 = open('albert2.txt', 'wb')
data3.write("They say Kim Jong Un is merely a puppet under Emperor Albert's control.")
data3.close()

data4 = open('albert2.txt', 'r')
print data4.read()
#data4 = data4.read()

#s3.Bucket('demo-alee156').put_object(Key='albert2.txt', Body=data4)

client.upload_file('albert2.txt', bucket, 'albert2.txt')
