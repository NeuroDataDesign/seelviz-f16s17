![https://github.com/NeuroDataDesign/seelviz/blob/gh-pages/AlbertImages/BatchFlow.pdf]()

Started out with a docker image with my administrator AWS credentials configured.
- Reads from s3 bucket
- Writes to file
- Changes file
- Uploads new file to s3 bucket

Ideally the workflow is
- Two separate docker images
  - One for web service, one of processing
- One ec2 instance that is lightweight nd is continuously running
  - Only needs to be powerful enough to upload to s3
- Batch instantly launches more powerful ec2 instance as soon as an object is uploaded to s3
  - Processing power is much more powerful
  
Logic for Seelviz:
- Webservice launched with lightweight ec2
- User types in a string token which is uploaded to s3
- This is trigger for lambda function
- Function submits job to batch which starts an ec2 instance
- Docker container created and instantly launched
- entrypoint to docker runs python script that automatically runs our pipeline
- output zipfile is pushed to s3 (different bucket than the first)
- this is trigger for webservice to output zipfile for download
  
Questions:
- Any way to bypass AWS credential requirements (don't know how secure it is to be logged on)
  - Should I create a guest credential? 
    - What permissions would it require, is it okay if I just give it read and write access to s3 and batch?
  - Need help with aws credentials csv when making Dockerfile (how to encrypt)
