import os
import streamlit as st
from groq import Groq
from dotenv import load_dotenv
import mysql.connector
import pandas as pd
from fpdf import FPDF

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ------- Prompt -------
PROMPT = """
You are an expert at converting English questions to MySQL queries.
The database is the Sakila sample database. Use these tables:

- film(film_id, title, description, release_year, rental_rate, rating, length, replacement_cost)
- actor(actor_id, first_name, last_name)
- film_actor(actor_id, film_id)
- customer(customer_id, first_name, last_name, email, active, store_id)
- rental(rental_id, customer_id, inventory_id, rental_date, return_date, staff_id)
- inventory(inventory_id, film_id, store_id)
- category(category_id, name)
- film_category(film_id, category_id)
- payment(payment_id, customer_id, amount, payment_date)

IMPORTANT MySQL rules you must always follow:
1. Never use COUNT() or any aggregate function in a WHERE clause. Use HAVING instead.
2. Never use LIMIT inside a subquery used with IN/ANY/ALL. Use a JOIN or derived table instead.
3. When joining film_category and category, always alias them separately. 
   Use: JOIN film_category fc ON f.film_id = fc.film_id JOIN category c ON fc.category_id = c.category_id
   Then SELECT c.name NOT fc.name.
4. Always use DISTINCT when joining inventory to avoid duplicate film rows.
5. For date filtering use: DATE_FORMAT(rental_date, '%Y-%m') = '2005-08' or MONTH()/YEAR() functions.
6. Always GROUP BY before HAVING.
7. Return ONLY the raw SQL. No markdown, no backticks, no explanation.

Examples:

"Show top 10 most rented films with category"
SELECT f.title, c.name AS category, COUNT(r.rental_id) AS rental_count
FROM film f
JOIN film_category fc ON f.film_id = fc.film_id
JOIN category c ON fc.category_id = c.category_id
JOIN inventory i ON f.film_id = i.film_id
JOIN rental r ON i.inventory_id = r.inventory_id
GROUP BY f.film_id, f.title, c.name
ORDER BY rental_count DESC
LIMIT 10;

"Which actors have appeared in more than 30 films?"
SELECT a.first_name, a.last_name, COUNT(fa.film_id) AS film_count
FROM actor a
JOIN film_actor fa ON a.actor_id = fa.actor_id
GROUP BY a.actor_id, a.first_name, a.last_name
HAVING COUNT(fa.film_id) > 30
ORDER BY film_count DESC;

"List the 5 longest unique films available in store 1"
SELECT DISTINCT f.title, f.length
FROM film f
JOIN inventory i ON f.film_id = i.film_id
WHERE i.store_id = 1
ORDER BY f.length DESC
LIMIT 5;

"Show total revenue per film category"
SELECT c.name AS category, SUM(p.amount) AS total_revenue
FROM payment p
JOIN rental r ON p.rental_id = r.rental_id
JOIN inventory i ON r.inventory_id = i.inventory_id
JOIN film_category fc ON i.film_id = fc.film_id
JOIN category c ON fc.category_id = c.category_id
GROUP BY c.name
ORDER BY total_revenue DESC;

"List customers who have spent more than 100 dollars"
SELECT c.first_name, c.last_name, SUM(p.amount) AS total_spent
FROM customer c
JOIN payment p ON c.customer_id = p.customer_id
GROUP BY c.customer_id, c.first_name, c.last_name
HAVING SUM(p.amount) > 100
ORDER BY total_spent DESC;

"Show films rented in July 2005 but not in August 2005"
SELECT DISTINCT f.title
FROM film f
JOIN inventory i ON f.film_id = i.film_id
JOIN rental r ON i.inventory_id = r.inventory_id
WHERE MONTH(r.rental_date) = 7 AND YEAR(r.rental_date) = 2005
AND f.film_id NOT IN (
    SELECT DISTINCT i2.film_id
    FROM inventory i2
    JOIN rental r2 ON i2.inventory_id = r2.inventory_id
    WHERE MONTH(r2.rental_date) = 8 AND YEAR(r2.rental_date) = 2005
);

"Which customers rented a film but never returned it?"
SELECT DISTINCT c.first_name, c.last_name, c.email
FROM customer c
JOIN rental r ON c.customer_id = r.customer_id
WHERE r.return_date IS NULL;

"How many rentals happened in August 2005?"
SELECT COUNT(*) AS total_rentals
FROM rental
WHERE MONTH(rental_date) = 8 AND YEAR(rental_date) = 2005;

"Show month wise total payment collected in 2005"
SELECT DATE_FORMAT(payment_date, '%Y-%m') AS month, SUM(amount) AS total_revenue
FROM payment
WHERE YEAR(payment_date) = 2005
GROUP BY DATE_FORMAT(payment_date, '%Y-%m')
ORDER BY month;

"Find all actors who have never acted in a horror film"
SELECT a.first_name, a.last_name
FROM actor a
WHERE a.actor_id NOT IN (
    SELECT fa.actor_id
    FROM film_actor fa
    JOIN film_category fc ON fa.film_id = fc.film_id
    JOIN category c ON fc.category_id = c.category_id
    WHERE c.name = 'Horror'
);

"Which store has generated more total revenue?"
SELECT i.store_id, SUM(p.amount) AS total_revenue
FROM payment p
JOIN rental r ON p.rental_id = r.rental_id
JOIN inventory i ON r.inventory_id = i.inventory_id
GROUP BY i.store_id
ORDER BY total_revenue DESC;

"Find the average rental rate of films grouped by rating"
SELECT rating, ROUND(AVG(rental_rate), 2) AS avg_rental_rate
FROM film
GROUP BY rating
ORDER BY avg_rental_rate DESC;

"List customers who rented films on the same day more than once"
SELECT c.first_name, c.last_name, DATE(r.rental_date) AS rental_day, COUNT(*) AS rentals
FROM customer c
JOIN rental r ON c.customer_id = r.customer_id
GROUP BY c.customer_id, c.first_name, c.last_name, DATE(r.rental_date)
HAVING COUNT(*) > 1
ORDER BY rentals DESC;

"Which film has the highest number of actors?"
SELECT f.title, COUNT(fa.actor_id) AS actor_count
FROM film f
JOIN film_actor fa ON f.film_id = fa.film_id
GROUP BY f.film_id, f.title
ORDER BY actor_count DESC
LIMIT 1;
"""

# ------- Groq call -------
def get_groq_response(question):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": PROMPT},
            {"role": "user",   "content": question},
        ],
    )
    return response.choices[0].message.content.strip()

# ------- MySQL query — fixed: now returns both columns AND rows -------
def run_sql_query(sql):
    conn = mysql.connector.connect(
        host=os.getenv("DB_HOST", "localhost"),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", ""),
        database="sakila"
    )
    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    columns = [desc[0] for desc in cur.description]  # ← was missing from return
    conn.close()
    return columns, rows  # ← now returns a tuple of (columns, rows)

# ------- PDF export -------
def export_to_pdf(sql, columns, rows, filename="query_results.pdf"):
    # Clean text to remove characters outside latin-1 range
    def safe(text):
        return str(text).encode("latin-1", errors="replace").decode("latin-1")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 10, safe("Text to SQL — Query Results"), ln=True, align="C")
    pdf.ln(4)

    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 8, safe("Generated SQL Query:"), ln=True)
    pdf.set_font("Courier", size=10)
    pdf.multi_cell(0, 7, safe(sql))
    pdf.ln(4)

    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 8, safe("Results:"), ln=True)

    # Column headers
    if columns:
        pdf.set_fill_color(220, 220, 240)
        pdf.set_font("Arial", "B", 9)
        for col in columns:
            pdf.cell(40, 7, safe(col), border=1, fill=True)
        pdf.ln()

    # Data rows
    pdf.set_font("Arial", size=9)
    for row in rows:
        for val in row:
            pdf.cell(40, 7, safe(str(val))[:20], border=1)
        pdf.ln()

    pdf.output(filename)
    return filename

# ------- Streamlit UI -------
st.set_page_config(page_title="Text to SQL — Groq")
st.title("Text → SQL with Groq AI")
st.caption("Querying the Sakila MySQL database")

question = st.text_input("Ask a question about the Sakila database:")

if st.button("Ask"):
    if question:
        with st.spinner("Generating SQL..."):
            sql = get_groq_response(question)

        st.subheader("Generated SQL")
        st.code(sql, language="sql")

        try:
            with st.spinner("Querying database..."):
                columns, rows = run_sql_query(sql)  # ← now correctly unpacks both

            st.subheader("Results")
            if rows:
                df = pd.DataFrame(rows, columns=columns)
                st.dataframe(df)

                # PDF download
                pdf_file = export_to_pdf(sql, columns, rows)
                with open(pdf_file, "rb") as f:
                    st.download_button("⬇ Download Results as PDF", f, file_name="query_results.pdf")
            else:
                st.info("Query returned no results.")

        except Exception as e:
            st.error(f"Database error: {e}")
    else:
        st.warning("Please enter a question first.")