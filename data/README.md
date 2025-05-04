**Download Flickr8k Dataset:**

   - Download the image ZIP file from [Flickr8K](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip) and unzip it. After unzipping, move all contents of the `Flickr8k_Dataset/` folder into the `data/` folder.

     **If you are using macOS or Linux, run:**

     ```bash
     unzip Flickr8k_Dataset.zip && mv Flickr8k_Dataset/* data/
     ```

     **If you are using Windows PowerShell, run:**

     ```powershell
     Expand-Archive -Path Flickr8k_Dataset.zip -DestinationPath .
     Move-Item -Path .\Flickr8k_Dataset\* -Destination .\data\
     ```

4. **Prepare the Data:**

   - Next, the Flickr8k images must be split into traning and validation sets. Run the following command to generate the training and validation splits:
     ```bash
     python code/data_prep.py
     ```
