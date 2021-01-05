import sqlite3
import os

conn = sqlite3.connect('reviews.sqlite')
c = conn.cursor()

c.execute('drop table if exists review_db')
c.execute('create table review_db'\
          '(review text, sentiment integer, date text)')


example1 = 'I love this movie, Cage is my favorite actor'
c.execute('insert into review_db'\
          '(review, sentiment, date) values'\
         '(?, ?, datetime("now"))', (example1, 1))

example2 = 'I dont like the following movies, they are boring and start late at night'
c.execute('insert into review_db'\
          '(review, sentiment, date) values'\
         '(?, ?, datetime("now"))', (example2, 0))

conn.commit()
conn.close()


conn = sqlite3.connect('reviews.sqlite')
c = conn.cursor()
c.execute('select * from review_db where date'\
         " between '2017-01-01' and datetime('now')")

results = c.fetchall()
conn.close()
print(results)



