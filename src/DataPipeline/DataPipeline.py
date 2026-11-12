import asyncio
import aiohttp
import websockets
import ssl
import certifi
import json
import csv
import os

from pathlib import Path
from datetime import datetime, timezone

# Environment variables and configuration constants
WSS_URL = os.getenv("WSS_URL")
RPC_URL = os.getenv("RPC_URL")
RAYDIUM_PROGRAM_ID = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"
COMMITMENT = "processed"

# File paths for raw logs and processed CSV features
DATA_DIR = Path("../../data")
RAW_FILE = DATA_DIR / "raw_transactions.jsonl"
CSV_FILE = DATA_DIR / "transaction_features.csv"

# SSL context using certified certificates
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())

RETRY_ATTEMPTS = 5
RETRY_DELAY = 1.0

# Define CSV columns for transaction metrics
CSV_FIELDS = [
    "signature",
    "slot",
    "notification_time",
    "enrichment_time",
    "enrichment_commitment",
    "notification_error",
    "transaction_error",
    "block_time",
    "compute_units",
    "total_fee_lamports",
    "account_count",
    "instruction_count",
    "inner_instruction_count",
    "program_ids",
    "token_mints",
    "token_balance_change_count",
    "token_balance_changes",
    "account_keys",
]


def initialize_files():
    #Ensure data directories and CSV headers exist before running.
    DATA_DIR.mkdir(exist_ok=True)
    if not CSV_FILE.exists() or CSV_FILE.stat().st_size == 0:
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
            writer.writeheader()

    print(f"Raw data file: {RAW_FILE}")
    print(f"CSV feature file: {CSV_FILE}")


def utc_now():
    # Return current UTC timestamp in ISO format.
    return datetime.now(timezone.utc).isoformat()


def write_jsonl(data):
    # Append a raw or enriched transaction record to the JSONL file.
    with open(RAW_FILE, "a", encoding="utf-8") as file:
        file.write(json.dumps(data) + "\n")


def write_csv(row):
    #Add parsed transaction features to the CSV file.
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writerow(row)


async def get_transaction(session, signature):
    #get full transaction metadata from the Solana RPC provider with retry logic.
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getTransaction",
        "params": [
            signature,
            {
                "encoding": "json",
                "commitment": "confirmed",
                "maxSupportedTransactionVersion": 1,
            },
        ],
    }

    for attempt in range(RETRY_ATTEMPTS):
        try:
            async with session.post(RPC_URL, json=payload, ssl=SSL_CONTEXT) as response:
                # Handle rate limiting (HTTP 429)
                if response.status == 429:
                    retry_after = response.headers.get("Retry-After")
                    wait_time = float(retry_after) if retry_after else min(2 ** attempt, 30)
                    print(f"Rate limited. Retrying in {wait_time} seconds")
                    await asyncio.sleep(wait_time)
                    continue

                response.raise_for_status()
                data = await response.json()

                if data.get("error"):
                    print(f"RPC error: {data['error']}")
                    await asyncio.sleep(min(2 ** attempt, 30))
                    continue

                result = data.get("result")
                if result is not None:
                    return result

                print(f"Transaction not available yet: {signature}")

        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as error:
            print(f"RPC request failed: {error}")

        if attempt < RETRY_ATTEMPTS - 1:
            await asyncio.sleep(min(2 ** attempt, 30))
    return None


def extract_features(tx, notification):
    """Extract and flatten structured features from transaction metadata and logs."""
    transaction = tx.get("transaction", {})
    message = transaction.get("message", {})
    meta = tx.get("meta") or {}

    # Normalize account keys into a list of strings
    account_keys = message.get("accountKeys", [])
    normalized_keys = []
    for key in account_keys:
        if isinstance(key, dict):
            normalized_keys.append(key.get("pubkey", ""))
        else:
            normalized_keys.append(str(key))

    instructions = message.get("instructions", [])
    inner_instructions = meta.get("innerInstructions", [])
    inner_instruction_count = sum(
        len(group.get("instructions", [])) for group in inner_instructions
    )

    # Collect program IDs invoked by instructions
    program_ids = []
    for instruction in instructions:
        program_id = instruction.get("programId")
        if program_id:
            program_ids.append(program_id)
        else:
            program_index = instruction.get("programIdIndex")
            if isinstance(program_index, int) and program_index < len(normalized_keys):
                program_ids.append(normalized_keys[program_index])

    # Parse pre and post token balances
    pre_balances = meta.get("preTokenBalances", [])
    post_balances = meta.get("postTokenBalances", [])

    pre_amounts = {}
    post_amounts = {}
    token_mints = set()

    for balance in pre_balances:
        mint = balance.get("mint")
        owner = balance.get("owner", "")
        account_index = balance.get("accountIndex")
        key = (account_index, mint, owner)
        amount = int(balance.get("uiTokenAmount", {}).get("amount", 0))
        pre_amounts[key] = amount
        if mint:
            token_mints.add(mint)

    for balance in post_balances:
        mint = balance.get("mint")
        owner = balance.get("owner", "")
        account_index = balance.get("accountIndex")
        key = (account_index, mint, owner)
        amount = int(balance.get("uiTokenAmount", {}).get("amount", 0))
        post_amounts[key] = amount
        if mint:
            token_mints.add(mint)

    # Calculate token balance changes across accounts
    all_balance_keys = set(pre_amounts.keys()) | set(post_amounts.keys())
    token_balance_changes = {}

    for key in all_balance_keys:
        before = pre_amounts.get(key, 0)
        after = post_amounts.get(key, 0)
        difference = after - before
        if difference != 0:
            account_index, mint, owner = key
            change_key = f"{account_index}:{mint}:{owner}"
            token_balance_changes[change_key] = difference

    notification_value = notification.get("params", {}).get("result", {}).get("value", {})
    context = notification.get("params", {}).get("result", {}).get("context", {})

    # Build final row dictionary for CSV output
    row = {
        "signature": transaction.get("signatures", [""])[0],
        "slot": context.get("slot", tx.get("slot", "")),
        "notification_time": notification_value.get("_received_at", ""),
        "enrichment_time": utc_now(),
        "enrichment_commitment": "confirmed",
        "notification_error": json.dumps(notification_value.get("err")),
        "transaction_error": json.dumps(meta.get("err")),
        "block_time": tx.get("blockTime"),
        "compute_units": meta.get("computeUnitsConsumed"),
        "total_fee_lamports": meta.get("fee"),
        "account_count": len(normalized_keys),
        "instruction_count": len(instructions),
        "inner_instruction_count": inner_instruction_count,
        "program_ids": json.dumps(list(set(program_ids))),
        "token_mints": json.dumps(sorted(token_mints)),
        "token_balance_change_count": len(token_balance_changes),
        "token_balance_changes": json.dumps(token_balance_changes),
        "account_keys": json.dumps(normalized_keys),
    }

    return row


async def stream_transactions():
    #Connect to the Solana WebSocket logs feed and process incoming Raydium transactions. Raydium is a DEX and AMM
    if not WSS_URL:
        raise ValueError("WSS_URL environment variable is missing.")
    if not RPC_URL:
        raise ValueError("RPC_URL environment variable is missing.")

    initialize_files()
    processed_signatures = set()

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
        async with websockets.connect(
            WSS_URL, ssl=SSL_CONTEXT, ping_interval=20, ping_timeout=20
        ) as websocket:
            print("\nConnected to Solana WebSocket.")

            # Subscribe to logs mentioning the Raydium program ID
            subscribe_payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "logsSubscribe",
                "params": [
                    {"mentions": [RAYDIUM_PROGRAM_ID]},
                    {"commitment": COMMITMENT},
                ],
            }

            await websocket.send(json.dumps(subscribe_payload))
            subscription_response = json.loads(await websocket.recv())
            print("Subscription response:", subscription_response)
            print("\nListening for transactions...\n")

            while True:
                message = await websocket.recv()
                notification = json.loads(message)

                # Filter for log notifications only
                if notification.get("method") != "logsNotification":
                    continue

                params = notification.get("params", {})
                result = params.get("result", {})
                context = result.get("context", {})
                value = result.get("value", {})

                signature = value.get("signature")
                slot = context.get("slot")

                if not signature:
                    continue

                # Avoid processing duplicate signatures
                if signature in processed_signatures:
                    continue

                processed_signatures.add(signature)
                notification_time = utc_now()
                value["_received_at"] = notification_time

                print("New transaction notification")
                print("Signature:", signature)
                print("Slot:", slot)
                print("Error:", value.get("err"))

                # Log raw notification
                write_jsonl({"type": "notification", "data": notification})

                # Fetch full transaction details via RPC
                tx_result = await get_transaction(session, signature)

                if tx_result is None:
                    print("Could not retrieve transaction metadata.")
                    write_jsonl({
                        "type": "enrichment_failed",
                        "signature": signature,
                        "slot": slot,
                        "time": utc_now(),
                    })
                    continue

                # Log enriched transaction payload
                write_jsonl({"type": "enriched_transaction", "signature": signature, "data": tx_result})

                # Extract features and save to CSV
                features = extract_features(tx_result, notification)
                write_csv(features)

                print("\nTransaction enriched and saved.")
                print("Compute units:", features["compute_units"])
                print("Fee:", features["total_fee_lamports"])
                print("Accounts:", features["account_count"])
                print("Instructions:", features["instruction_count"])
                print("Token balance changes:", features["token_balance_change_count"])
                print(f"CSV updated: {CSV_FILE}")


async def main():
    #Main entry point for running the async transaction streaming pipeline.
    try:
        await stream_transactions()
    except KeyboardInterrupt:
        print("\nStopping transaction collector.")
    except Exception as error:
        print(f"\nPipeline error: {error}")


if __name__ == "__main__":
    asyncio.run(main())