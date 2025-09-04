## Setting Up Google Drive Access via Service Account (PyDrive2)

You'll need a service account to fetch the WireSegHR dataset using a script in this folder. Follow the steps below:

1. **Create a Service Account**  
   - Navigate to the Google Cloud Console. Under **IAM & Admin → Service Accounts**, create a new service account.  
   - Assign it a Viewer role.

2. **Generate and Download JSON Key**  
   - In the service account details, go to **Keys → Add Key → Create new key**, choose **JSON**, and download the key file.  
   - Save this file locally to this repo as `secrets/drive-json.json`.

3. **Share the Drive Folder or Files**  
   - Grant the service account access to the target Drive folder - https://drive.google.com/drive/folders/1fgy3wn_yuHEeMNbfiHNVl1-jEdYOfu6p - using its service account email. Go to the folder in Google Drive, click on the share button, and add the service account email with Viewer permissions.


