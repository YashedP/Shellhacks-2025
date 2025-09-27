import ee

# The service account email (from "client_email" field or in IAM console)
SERVICE_ACCOUNT = 'shellhacks@striped-orbit-473405-h0.iam.gserviceaccount.com'
KEY_FILE = 'gcp_key.json'
PROJECT = 'striped-orbit-473405-h0'   # the GCP project you registered with Earth Engine

# Create credentials and initialize
credentials = ee.ServiceAccountCredentials(SERVICE_ACCOUNT, KEY_FILE)
ee.Initialize(credentials, project=PROJECT)

# quick test: load Florida boundary and print a count
florida = ee.FeatureCollection("TIGER/2018/States").filter(ee.Filter.eq('STUSPS', 'FL'))
print('does api work', "yes" if 1 == florida.size().getInfo() else "no")
