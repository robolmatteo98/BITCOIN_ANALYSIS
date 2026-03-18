DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'fk_region_address'
    ) THEN
        ALTER TABLE address
        ADD CONSTRAINT FK_region_address
        FOREIGN KEY (region_id) REFERENCES region (id);
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_transaction_txid
ON transaction (txid);

CREATE INDEX IF NOT EXISTS idx_transaction_block
ON transaction (fk_block_id);

CREATE INDEX IF NOT EXISTS idx_tx_input_transaction
ON tx_input (fk_transaction_id);

CREATE INDEX IF NOT EXISTS idx_tx_input_prev_outpoint
ON tx_input (prev_transaction_id, prev_vout);

CREATE INDEX IF NOT EXISTS idx_tx_output_tx_n
ON tx_output (fk_transaction_id, n);

CREATE INDEX IF NOT EXISTS idx_tx_output_address
ON tx_output (fk_address_code);

CREATE INDEX IF NOT EXISTS idx_block_time
ON block (time);
