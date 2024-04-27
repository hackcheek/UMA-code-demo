# Use the latest foundry image
FROM ghcr.io/foundry-rs/foundry AS foundry

# Fill docker compose volume :app
COPY ./scripts/ /app/scripts
COPY ./contracts/ /app/contracts/
COPY ./freewillai/ /app/freewillai/
COPY ./lib/ /app/lib/

EXPOSE 8545
ENTRYPOINT  ["anvil", "--host", "0.0.0.0", "--port", "8545","--config-out", "/anvil/configs.json"]
