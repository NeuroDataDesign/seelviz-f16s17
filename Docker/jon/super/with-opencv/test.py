import clarityviz as cl

token = 's275_to_ara3'
cert_path = '../userToken.pem'
# cl.analysis.get_registered(token, cert_path)

cl.analysis.run_pipeline(token, cert_path, 5)



