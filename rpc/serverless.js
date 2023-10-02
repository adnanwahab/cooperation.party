module.exports.read = async event => {
  console.log('Starting read function');
  if(!event || !event.queryStringParameters || !event.queryStringParameters.cells) {
    return formatResponse(400, { message: 'Invalid parameters' });
  }

  const cells = event.queryStringParameters.cells;

  await spreadsheetAuth(doc);
  await doc.loadInfo();
  const sheet = doc.sheetsByIndex[0];
  await sheet.loadCells();
  const contents = cells.split(',').map(cell => sheet.getCellByA1(cell).value);
  return formatResponse(200, { contents });
};
//https://dev.to/a0viedo/writing-to-a-google-sheet-using-serverless-2ndc
// import gspread
// from oauth2client.service_account import ServiceAccountCredentials
// def write_to_google_sheets(data):
//     # Set up Google Sheets API credentials
//     scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/spreadsheets",
//              "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]
//     credentials = ServiceAccountCredentials.from_json_keyfile_name("your-credentials.json", scope)
//     client = gspread.authorize(credentials)
//     # Replace 'your-spreadsheet-id' with your actual Google Sheets spreadsheet ID
//     spreadsheet = client.open_by_key("your-spreadsheet-id")
//     # Select the specific worksheet within the spreadsheet
//     worksheet = spreadsheet.get_worksheet(0)
//     # Append the data to the worksheet
//     worksheet.append_row(data)
// # Example data to write
// data_to_write = ["John", "Doe", "johndoe@example.com"]
// # Call the function to write the data
// write_to_google_sheets(data_to_write)
const { google } = require('googleapis');
const sheets = google.sheets('v4');

async function writeToGoogleSheets(auth, data) {
  const sheetsAPI = google.sheets({ version: 'v4', auth });

  const spreadsheetId = 'your-spreadsheet-id'; // Replace with your actual spreadsheet ID
  const range = 'Sheet1!A:F'; // Replace with the desired sheet name and range

  try {
    const response = await sheetsAPI.spreadsheets.values.append({
      spreadsheetId,
      range,
      valueInputOption: 'RAW',
      insertDataOption: 'INSERT_ROWS',
      resource: {
        values: [data],
      },
    });

    console.log('Data appended successfully:', response.data);
  } catch (err) {
    console.error('Error appending data:', err);
  }
}

// Load the credentials JSON file you downloaded
const credentials = require('./your-credentials.json');

// Example data to write
const dataToWrite = ['John', 'Doe', 'johndoe@example.com'];

// Set up Google API credentials
const { client_secret, client_id, redirect_uris } = credentials.installed;
const auth = new google.auth.OAuth2(client_id, client_secret, redirect_uris[0]);
auth.setCredentials(credentials.tokens);

// Call the function to write the data
