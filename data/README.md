# Dados

`raw/telco_customer_churn.csv` é o dataset público **Telco Customer Churn**, disponibilizado pela IBM no repositório [IBM/telco-customer-churn-on-icp4d](https://github.com/IBM/telco-customer-churn-on-icp4d) (licença Apache 2.0).

- 7.043 clientes de uma operadora de telecom fictícia
- 21 colunas: perfil demográfico, serviços contratados, tipo de contrato, forma de pagamento, valores cobrados e se o cliente cancelou (`Churn`)
- Valores monetários em dólares (US$)

O arquivo é mantido sem alterações. Toda limpeza acontece em código (`src/churn/data.py`), para que o processo seja reproduzível.
