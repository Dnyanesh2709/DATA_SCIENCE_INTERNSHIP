CREATE DATABASE DAY2;
USE DAY2;
-- CONSTRAINTS
-- UNIQUE AND NOT NULL CONSTRAINTS
CREATE TABLE emp(
empID int NOT NULL,
empName VARCHAR (50) UNIQUE KEY,
country VARCHAR(50)
);
INSERT INTO EMP(empId,empName,country)
VALUES (100,'XYZ','INDIA');

-- FOREIGN KEY
CREATE TABLE Customers(
customerID INT PRIMARY KEY,
customerName VARCHAR(50) NOT NULL
);

CREATE TABLE Orders(
orderID int PRIMARY KEY,
orderNUMBER INT NOT NULL,
customerID INT,
FOREIGN KEY (customerID) REFERENCES Customers (customerID)
);
INSERT INTO Customers (customerID,customername)
VALUES (1,'John'),(2,'Jane'),(3,'Bob');

INSERT INTO Orders (orderID,orderNUMBER,customerID)
VALUES (1,101,1),(2,102,2),(3,103,3);

SELECT * FROM Customers;
SELECT * FROM Orders;

-- INSERT INTO Orders(orderID,orderNUMBER,customerID)
-- VALUES (4,104,4);
#cannot add value in child table unless we add a corresponding value in the master table, similarly, we cannot perform deletion in the master table unless we delete it's corresponding values from child table
#DELETE FROM Customers
#WHERE customerID='3';

#set operations
#UNION/UNION ALL/INTERSECT/EXCEPT

-- Create the first table
CREATE TABLE Employees (
    EmpID INT,
    Name VARCHAR(100),
    Country VARCHAR(100)
);

-- Insert data into the first table
INSERT INTO Employees (EmpID, Name, Country) VALUES
(1, 'Shubham', 'India'),
(2, 'Aman', 'Australia'),
(3, 'Naveen', 'Sri Lanka'),
(4, 'Aditya', 'Austria'),
(5, 'Nihant', 'Spain');

-- Create the second table
CREATE TABLE Employees2 (
    EmpID INT,
    Name VARCHAR(100),
    Country VARCHAR(100)
);

-- Insert data into the second table
INSERT INTO Employees2 (EmpID, Name, Country) VALUES
(1, 'Tommy', 'England'),
(2, 'Allen', 'France'),
(3, 'Nancy', 'India'),
(4, 'Adi', 'Ireland'),
(5, 'Sandy', 'Spain');

-- SQL query to combine both tables and remove duplicates
#UNION
SELECT Country FROM Employees
UNION
SELECT Country FROM Employees2
ORDER BY Country;

#UNION ALL
SELECT Country FROM Employees
UNION ALL
SELECT Country FROM Employees2
ORDER BY Country;

SELECT Country FROM Employees
INTERSECT
SELECT Country FROM Employees2
ORDER BY Country;

SELECT Country FROM Employees
EXCEPT
SELECT Country FROM Employees2
ORDER BY Country;

-- SQL JOINS
#GENERALIZED SYNTAX
#SELECT COLUMN_NAME(S)
#FROM TableA
#INNER JOIN TableB
#ON TableA.col_name=TableB.col_name


-- Create the customer table
CREATE TABLE customer (
    Customer_id INT,
    First_name VARCHAR(100),
    Last_name VARCHAR(100),
    Address_id INT
);


-- Insert data into the customer table
INSERT INTO customer (Customer_id, First_name, Last_name, Address_id) VALUES
(1, 'Mary', 'Jonas', 5),
(3, 'Linda', 'Smith', 7),
(4, 'Madan', 'Mohan', 8),
(2, 'Bary', 'Williams', 6);

-- Create the payment table
CREATE TABLE payment (
    Customer_id INT,
    Amount INT,
    Mode VARCHAR(100),
    Date DATE
);

-- Insert data into the payment table
INSERT INTO payment (Customer_id, Amount, Mode, Date) VALUES
(1, 60, 'Cash', '2024-04-03'),
(2, 30, 'Credit card', '2024-04-05'),
(3, 110, 'Cash', '2024-04-15'),
(10, 70, 'Mobile Payment', '2024-04-20'),
(11, 80, 'Cash', '2024-04-29');

-- INNER JOIN
SELECT *
FROM customer AS c
INNER JOIN payment AS p
ON c.customer_id=p.customer_id;

-- LEFT JOIN
SELECT *
FROM customer AS c
LEFT JOIN payment AS p
on c.customer_id=p.customer_id;

-- RIGHT JOIN
SELECT *
FROM customer AS c
 JOIN payment AS p
on c.customer_id=p.customer_id;

#WINDOW FUNCTIONS
#SYNTAX
#SELECT COLUMN_NAMES,
-- ( [<PARTITION BY CLAUSE>]
-- [<ORDER BY CLAUSE>]
-- [<ROW OR RANGE CLAUSE>] 

-- Create the table
CREATE TABLE example_table (
    new_id INT,
    new_cat VARCHAR(50)
);

-- Insert the data into the table
INSERT INTO example_table (new_id, new_cat) VALUES
(100, 'Agni'),
(200, 'Agni'),
(500, 'Dharti'),
(700, 'Dharti'),
(200, 'Vayu'),
(300, 'Vayu'),
(500, 'Vayu');

SELECT * FROM example_table;
SELECT new_id,new_cat,
SUM(new_id) OVER( partition by new_cat ORDER BY new_id) AS "TOTAL",
avg(new_id) OVER( partition by new_cat order by new_id) AS "AVERAGE",
count(new_id) OVER( partition by new_cat order by new_id) AS "COUNT",
min(new_id) OVER( partition by new_cat order by new_id) AS "MINIMUM",
max(new_id) OVER( partition by new_cat order by new_id) AS "MAXIMUM"
FROM example_table;

SELECT new_id,new_cat,
SUM(new_id)OVER( order by new_id ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS "TOTAL",
avg(new_id)OVER( order by new_id ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS "AVERAGE",
COUNT(new_id)OVER( order by new_id ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS "COUNT",
min(new_id)OVER( order by new_id ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS "MINIMUM",
max(new_id)OVER( order by new_id ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS "MAXIMUM"
FROM example_table;


SELECT new_id,
ROW_NUMBER() over( order by new_id) AS "ROW_NUMBER",
rank() OVER( order by new_id) AS "RANK",
dense_rank() over( order by new_id) AS "DENSE_RANK",
percent_rank() over( order by new_id) AS "PERCENT_RANK"
FROM example_table;

-- ANALYTICS
SELECT new_id,
first_value(new_id) over( order by new_id) AS "FIRST_VALUES",
last_value(new_id) OVER( order by new_id) AS "LAST_VALUE",
lead(new_id) over( order by new_id) AS "LEAD",
lag(new_id) over( order by new_id) AS "LAG"
FROM example_table;
