CREATE OR REPLACE VIEW address_feature_summary AS
SELECT
    balance.address,
    balance.received_output_count,
    balance.spent_output_count,
    balance.unspent_output_count,
    balance.total_received,
    balance.total_sent,
    balance.net_flow,
    balance.estimated_balance,
    activity.first_seen,
    activity.last_seen,
    activity.active_span,
    activity.activity_count,
    activity.active_days,
    a.region_id
FROM address_balance_summary balance
LEFT JOIN address_activity_summary activity
  ON activity.address = balance.address
LEFT JOIN address a
  ON a.code = balance.address;

-- mette insieme parte economica, parte temporale e region_id. Possibile utilizzarla come base per feature engineering o per esportare dataset.