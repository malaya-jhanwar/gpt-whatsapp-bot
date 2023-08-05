"""API endpoints for managing whatsapp weebhooks."""
from datetime import datetime
from decimal import Decimal
import json
import logging
import os
import re
from typing import Final
import uuid

from dotenv import load_dotenv
# from essentials.basemodels import ChatOutput
# from essentials.db import ChatGPTDatabaseConnector
# from essentials.db import DatabaseConnector
# from essentials.security import TokenAuth
from fastapi import APIRouter
from fastapi import Depends
from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts.chat import ChatPromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate
from langchain.schema import OutputParserException
import matplotlib.pyplot as plt
import pandas as pd
from psycopg2.errors import UndefinedTable
import pycurl
import pytz

router = APIRouter()

logger = logging.getLogger(__name__)
token_auth = TokenAuth()
load_dotenv()

WHATSAPP_API_TOKEN = os.getenv("WHATSAPP_API_TOKEN")
WHATSAPP_PHONE_ID = os.getenv("WHATSAPP_PHONE_ID")

MAX_ROWS_IN_SQL_QUERY: Final[int] = 100


class JailbreakAttemptException(Exception):
    """To be thrown if user is potentially attempting a jailbreak."""


def get_client_whatsapp_numbers(
    database_connector: DatabaseConnector,
    user_id: int,
) -> list[str]:
    """Get the client contact list for sending report."""
    query = """
    SELECT whatsapp FROM whatsapp_contacts WHERE user_id = %s
    """

    with database_connector() as connector:
        with connector.cursor() as cursor:
            cursor.execute(query, (user_id, ))
            data = cursor.fetchall()
            if not data:
                return []
            contact_list = list(map(lambda x: x[0][1:], data))
            return contact_list


def send_report_to_client(recipient: str, media_id: str, filename: str, start_date: str, end_date: str) -> int:
    """Send the report to the client."""
    curl = pycurl.Curl()

    # Set URL and options
    curl.setopt(pycurl.URL, f'https://graph.facebook.com/v15.0/{WHATSAPP_PHONE_ID}/messages')
    curl.setopt(pycurl.HTTPHEADER, [f'Authorization: Bearer {WHATSAPP_API_TOKEN}', 'Content-Type: application/json'])
    curl.setopt(pycurl.POST, 1)
    data = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": recipient,
        "type": "template",
        "template": {
            "name":
                "weekly_report",
            "language": {
                "code": "en"
            },
            "components": [{
                "type": "header",
                "parameters": [{
                    "type": "document",
                    "document": {
                        "id": media_id,
                        "filename": filename
                    }
                }]
            }, {
                "type": "body",
                "parameters": [{
                    "type": "text",
                    "text": start_date
                }, {
                    "type": "text",
                    "text": end_date
                }]
            }]
        }
    }

    # Modify data with dynamic variables
    curl.setopt(pycurl.POSTFIELDS, json.dumps(data))

    # Get response
    curl.perform()
    response_code = curl.getinfo(pycurl.RESPONSE_CODE)

    # Print response
    logger.debug(response_code)
    # Cleanup
    curl.close()
    return response_code


#
# ChatGPT automated chat section
#


def upload_image_and_get_media_id(path_to_image: str) -> str:
    """Upload the report to the WhatsApp API and get the media ID."""
    url = f'https://graph.facebook.com/v17.0/{WHATSAPP_PHONE_ID}/media'
    headers = [
        f'Authorization: Bearer {WHATSAPP_API_TOKEN}',
    ]
    fields = [
        ('file', (pycurl.FORM_FILE, str(path_to_image))),
        ('type', 'image/png'),
        ('messaging_product', 'whatsapp'),
    ]

    c = pycurl.Curl()
    c.setopt(pycurl.URL, url)
    c.setopt(pycurl.HTTPHEADER, headers)
    c.setopt(pycurl.HTTPPOST, fields)

    response = bytearray()
    c.setopt(pycurl.WRITEFUNCTION, response.extend)
    c.perform()
    c.close()

    # Convert the response to a JSON object
    json_data = json.loads(response.decode('utf-8'))

    try:
        media_id = json_data['id']
    except KeyError:
        logger.exception("Unable to send report. Check whether you have created the .env file.",
                         extra={"json_data": json_data})
        raise

    return media_id


def send_whatsapp_image(
    to_number: str,
    media_id: str,
) -> int:
    """Send a WhatsApp message with an image.

    Args:
        media_id (str): id of the image to send
        to_number (str): WhatsApp number to which to send the message

    Returns:
        Response code from pycurl or -1 if an error occurred
    """
    url = f"https://graph.facebook.com/v17.0/{WHATSAPP_PHONE_ID}/messages"
    headers = [
        f"Authorization: Bearer {WHATSAPP_API_TOKEN}",
        "Content-Type: application/json",
    ]
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to_number,
        "type": "image",
        "image": {
            "id": media_id
        }
    }

    curl = pycurl.Curl()
    curl.setopt(pycurl.URL, url)
    curl.setopt(pycurl.HTTPHEADER, headers)
    curl.setopt(pycurl.POST, 1)
    curl.setopt(pycurl.POSTFIELDS, json.dumps(payload))

    try:
        # Perform the request
        curl.perform()
        response_code = curl.getinfo(pycurl.RESPONSE_CODE)
        return response_code
    except pycurl.error:
        # Handle any errors
        logger.exception("Unable to send media message.")
    finally:
        # Cleanup
        curl.close()

    return -1


def send_whatsapp_message(to: str, message_content: str) -> int:
    url = f"https://graph.facebook.com/v17.0/{WHATSAPP_PHONE_ID}/messages"
    headers = [
        f"Authorization: Bearer {WHATSAPP_API_TOKEN}",
        "Content-Type: application/json",
    ]
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to,
        "type": "text",
        "text": {
            "preview_url": False,
            "body": message_content
        }
    }

    curl = pycurl.Curl()
    curl.setopt(pycurl.URL, url)
    curl.setopt(pycurl.HTTPHEADER, headers)
    curl.setopt(pycurl.POST, 1)
    curl.setopt(pycurl.POSTFIELDS, json.dumps(payload))

    try:
        # Perform the request
        curl.perform()
        response_code = curl.getinfo(pycurl.RESPONSE_CODE)
        return response_code
    except pycurl.error:
        logger.exception("A pycurl error occurred.")
    finally:
        # Cleanup
        curl.close()

    return -1


def generate_data_image(column_names, rows):
    """Generate a table image from the data in the chat GPT API response.

    Args:
        column_names (list): list of column names
        rows (list): list of rows

    Returns:
        str: path to the image
    """
    # Assuming 'result' contains the data from the chat GPT API response
    updated_rows = []
    for row in rows:
        updated_row = []
        for cell in row:
            if isinstance(cell, Decimal):
                updated_row.append(round(float(cell), 2))
            else:
                updated_row.append(cell)
        updated_rows.append(updated_row)

    df = pd.DataFrame(updated_rows, columns=column_names)
    logger.debug(df.dtypes)
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')
    mpl_table = ax.table(cellText=df.values,
                         colLabels=df.columns,
                         loc='center',
                         colColours=['#f5f5f5'] * len(df.columns))
    # make the numeric columns right aligned and rest left aligned

    for i in range(len(df.columns)):
        if df.columns[i] in numeric_columns:
            mpl_table[:, i].set_horizontalalignment('right')
        else:
            mpl_table[:, i].set_horizontalalignment('left')

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(10)

    # Add styling to the header row
    for i, col in enumerate(df.columns):
        mpl_table[(0, i)].set_facecolor('#f8f8f8')
        mpl_table[(0, i)].set_text_props(fontweight='bold')

    # Add borders to cells
    for i in range(len(df.index) + 1):
        for j in range(len(df.columns)):
            cell = mpl_table[(i, j)]
            cell.set_linewidth(0.5)
            cell.set_edgecolor('gray')

    # Add padding between cells
    plt.subplots_adjust(left=0.15, right=0.85)

    # Add cell coloring for alternating rows
    for i in range(1, len(df.index) + 1):
        if i % 2 == 0:
            for j in range(len(df.columns)):
                cell = mpl_table[(i, j)]
                cell.set_facecolor('#f9f9f9')

    # Add a title for the table
    table_title = "Response"
    plt.title(table_title, fontsize=14, fontweight='bold')

    # Save the table to a png file
    unique_name = f"/tmp/{uuid.uuid4().hex}.png"
    plt.savefig(fname=unique_name, format="png", bbox_inches='tight', pad_inches=0.5)
    return unique_name


# get user_id of the client from database
def get_user_id_given_requestor_number(requestor_number: str, database_connector: DatabaseConnector) -> int:
    """Gets the user id from the database.

    Args:
        requestor_number (str): WhatsApp number of the user who is requesting for data

    Returns:
        int: user id of the user
    """
    query = """
    SELECT user_id FROM whatsapp_contacts WHERE whatsapp = %s
    """
    with database_connector() as connector:
        with connector.cursor() as cursor:
            cursor.execute(query, ("+" + requestor_number, ))
            result = cursor.fetchone()
            if result is None:
                return -1
            user_id = result[0]
            logger.debug(user_id)
            return user_id


@lru_cache
def get_user_device_details(user_id: int) -> tuple[str, str]:
    """Get the categories and device names associated with the user."""
    database_connector = ChatGPTDatabaseConnector()
    with database_connector() as connector:
        with connector.cursor() as cursor:
            device_cat_query = """
            SELECT category
            FROM
                device_categories WHERE device_id IN
            (SELECT deviceid FROM deviceownership WHERE user_id = %s)
            """
            # fetch all device categories of the user
            cursor.execute(device_cat_query, (user_id, ))
            device_cat = cursor.fetchall()

            device_name_query = """SELECT device_nicename FROM deviceownership WHERE user_id = %s"""
            # fetch all device nicenames of the user
            cursor.execute(device_name_query, (user_id, ))
            device_names = cursor.fetchall()

            return ', '.join({cat[0] for cat in device_cat}), ', '.join({cat[0] for cat in device_names})


def exception_diagnostics(sender_number, input_prompt, chat_output):
    if input_prompt is not None:
        logger.debug("ChatGPT prompt", extra={"input_prompt": input_prompt})
    else:
        logger.debug("ChatGPT prompt", extra={"input_prompt": "Exception occurred before generating prompt."})
    if chat_output is not None:
        logger.debug("ChatGPT response", extra={"chat_output.content": chat_output.content})
    else:
        logger.debug("ChatGPT response",
                     extra={"chat_output.content": "Exception occurred before response was obtained."})
    send_whatsapp_message(sender_number, "Sorry, I could not understand your question. Could you please rephrase it?")


def ask_gpt(
    user_id: int,
    question: str,
    message_id: str,
    sender_number: str,
) -> bool:
    """Return answers for queries from the chat interface."""
    chat_output = None
    input_prompt = None
    logger.info("ChatGPT user prompt", extra={"user_id": user_id, "question": question, "requestor": sender_number})
    try:
        # prompt that provides the db info
        # - The values of columns deviceid or device_id must not be revealed. Instead, show only the device_nicename.
        template = """
            Answer the following questions as best as you can using the
            provided postgres table schema:

            - Table storing hourly usage: hourly_usage(time_frame_ist: timestamp, device_id: text, usage: numeric(watt
            hour))
            - Table storing device ownership details:
            deviceownership(deviceid: text, device_nicename: text, user_id: int, meter_type: text)
            - Devices are categorized using this table: device_categories(device_id: text, category: text)
            - Give a nice human readable alias to all columns

            Generated queries must abide by the below rules:
            - Device categories: {device_categories}
            - Device nicenames: {device_names}
            - Current timestamp: {current_time}
            - The deviceid or device_id must not be a part of the SELECT. Instead, you may SELECT device_nicename.
            - Unless the query is an aggregate, show either the date or the device nicename or both along with the usage
            column.
            - Want the usage in kilo watt hour make changes wherever necessary
            - If the question is to aggregate hourly_usage.usage of all devices, then only include devices whose
            meter_type = 'MAIN' and ignore rest.

            {format_instructions}
            """

        # Check if the user's question contains relevant keywords before proceeding
        relevant_keywords = [
            "usage", "devices", "categories", "owned", "last hour", "last day", "last week", "yesterday", "kwh",
            "electricity", "consumption", "consumed", "consumes", "device"
        ]
        is_relevant_question = any(keyword in question.lower() for keyword in relevant_keywords)

        if not is_relevant_question:
            # If the question is not relevant, do not ask Chat GPT for an answer.
            # Instead, respond with a message stating that the question is not relevant.
            alert_response = "Sorry, I can only answer questions related to data usage, devices, or categories."
            send_whatsapp_message(sender_number, alert_response)
            logger.error(alert_response)
            return False

        # question from the chat interface
        human_template = """Question: {question} for user id {user_id}
        (note: do not use 'do' alias for deviceownership)"""
        system_message_prompt = SystemMessagePromptTemplate.from_template(template)
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
        output_parser = PydanticOutputParser(pydantic_object=ChatOutput)  # type: ignore
        current_time = datetime.now(tz=pytz.timezone("Asia/Kolkata"))

        # final prompt(`template` + `human_template`) to input to the model
        device_categories, device_names = get_user_device_details(user_id)
        input_prompt = chat_prompt.format_prompt(
            device_categories=device_categories,
            device_names=device_names,
            current_time=str(current_time.replace(tzinfo=None)),
            format_instructions=output_parser.get_format_instructions(),
            question=question,
            user_id=user_id,
        ).to_messages()

        chat_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)  # type: ignore
        chat_output = chat_model(input_prompt)
        chat_output_content = output_parser.parse(chat_output.content)

        # sql query generated by the gpt model
        sql_query = chat_output_content.answer
        logger.debug("ChatGPT response", extra={"chat_output.content": chat_output.content})

        # add a limit so that we don't get exceedingly large results
        if sql_query[-1] == ";" and "LIMIT" not in sql_query:
            sql_query = sql_query[:-1]
            sql_query += f" LIMIT {MAX_ROWS_IN_SQL_QUERY}"

        pattern = rf"user_id = {user_id}"
        if re.search(pattern, sql_query) is None:
            raise JailbreakAttemptException("Possible jailbreak attempt!")

        database_connector = ChatGPTDatabaseConnector()
        with database_connector() as connector:
            connector.set_session(readonly=True)
            with connector.cursor() as cursor:
                cursor.itersize = MAX_ROWS_IN_SQL_QUERY
                cursor.execute(sql_query)
                rows = cursor.fetchmany(size=MAX_ROWS_IN_SQL_QUERY)

                column_names = [desc[0] for desc in cursor.description]

                # TODO if number of rows is exactly 100, it probably means that the result was truncated
                # TODO in this case, we should indicate somehow in the image that this list has been truncated
                image_path = generate_data_image(column_names, rows)
                image_media_id = upload_image_and_get_media_id(image_path)
                send_data_id = send_whatsapp_image(sender_number, image_media_id)
                return True if send_data_id != -1 else False

    except UndefinedTable:
        logger.exception("Table does not exist!")
        exception_diagnostics(sender_number, input_prompt, chat_output)
    except SyntaxError:
        logger.exception("Malformed query!")
        exception_diagnostics(sender_number, input_prompt, chat_output)
    except OutputParserException:
        logger.exception("Incorrectly formatted response!")
        exception_diagnostics(sender_number, input_prompt, chat_output)
    except JailbreakAttemptException:
        logger.exception("Potential jailbreak!", extra={"user_question": question})
        exception_diagnostics(sender_number, input_prompt, chat_output)
    except Exception:
        logger.exception("Critical!")
        exception_diagnostics(sender_number, input_prompt, chat_output)
    return False


@router.get("/user/whatsapp", tags=["Whatsapp"], response_class=HTMLResponse)
async def handle_whatsapp_get_hook(request: Request) -> str:
    """Webhook to handle whatsapp business API registration."""
    mode = request.query_params.get('hub.mode')
    challenge = request.query_params.get('hub.challenge')
    verify_token = request.query_params.get('hub.verify_token')
    if mode == "subscribe" and verify_token == os.getenv("WHATSAPP_WEBHOOK_VERIFY_TOKEN"):
        return str(challenge)
    return "Failure"


@router.post("/user/whatsapp", tags=["Whatsapp"], response_class=HTMLResponse)
async def handle_whatsapp_post_hook(
        request: Request,
        database_connector=Depends(DatabaseConnector),
) -> JSONResponse:
    """Webhook for receiving whatsapp messages and performing relevant actions."""
    message = await request.json()
    try:
        content = message["entry"][0]['changes'][0]['value']['messages'][0]['button']['payload']
        if content != 'noapprove':
            user_id, media_id, filename, start_date, end_date = content.split(';')

            contact = get_client_whatsapp_numbers(database_connector, user_id)
            # logger.debug(contact)

            for recipient in contact:
                response_code = send_report_to_client(recipient, media_id, filename, start_date, end_date)
                if response_code < 200 or response_code > 300:
                    logger.error("Could not send the report. Response code from curl = %s", response_code)

            return JSONResponse(content="Ok", status_code=200)
    except KeyError:
        pass

    try:
        content = message['entry'][0]['changes'][0]['value']['messages'][0]
        sender_number: str = content['from']
        message_body: str = content['text']['body']
        message_id: str = content['id']
        # user_id = get_user_id_given_requestor_number(sender_number, database_connector)
        user_id = 57
        if user_id != -1:
            ask_gpt(user_id, message_body, message_id, sender_number)
            return JSONResponse(content="Ok", status_code=200)
        else:
            logger.error("Unknown user asked a question or sent a message. Not forwarding it to ChatGPT.")
            return JSONResponse(content="Unknown user", status_code=200)

    except KeyError:
        logger.exception("Could not parse the message: %s", message)
        return JSONResponse(content="Could not parse the message", status_code=500)

    except Exception:
        logger.exception("Critical")
        return JSONResponse(content="Unknown exception", status_code=500)
