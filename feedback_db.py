"""
Feedback database for storing teacher corrections and ratings
Uses SQLite for lightweight local storage
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path


class FeedbackDB:
    """SQLite-based feedback database"""

    def __init__(self, db_path='feedback.db'):
        """
        Initialize feedback database

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self):
        """Create feedback table if it doesn't exist"""
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_id TEXT NOT NULL,
                student_id INTEGER NOT NULL,
                rating INTEGER NOT NULL CHECK(rating >= 1 AND rating <= 5),
                comments TEXT,
                corrections JSON,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(report_id, student_id)
            )
        ''')
        self.conn.commit()

    def add_feedback(self, report_id, student_id, rating, comments='', corrections=None):
        """
        Add or update feedback entry

        Args:
            report_id: Unique identifier for the report
            student_id: ID of the student
            rating: 1-5 star rating
            comments: Optional text feedback
            corrections: Optional dict of corrections (e.g., {"score": 0.3})

        Returns:
            True if successful
        """
        try:
            corrections_json = json.dumps(corrections or {})

            self.conn.execute('''
                INSERT OR REPLACE INTO feedback
                (report_id, student_id, rating, comments, corrections, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (report_id, student_id, rating, comments, corrections_json, datetime.now().isoformat()))

            self.conn.commit()
            return True

        except sqlite3.Error as e:
            print(f"Database error: {e}")
            return False

    def get_feedback_for_report(self, report_id):
        """
        Get all feedback entries for a report

        Args:
            report_id: Report identifier

        Returns:
            List of feedback records as dicts
        """
        cursor = self.conn.execute(
            'SELECT * FROM feedback WHERE report_id = ? ORDER BY timestamp DESC',
            (report_id,)
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_feedback_for_student(self, report_id, student_id):
        """
        Get feedback for specific student in specific report

        Args:
            report_id: Report identifier
            student_id: Student ID

        Returns:
            Feedback record as dict or None
        """
        cursor = self.conn.execute(
            'SELECT * FROM feedback WHERE report_id = ? AND student_id = ?',
            (report_id, student_id)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_all_feedback(self, limit=100):
        """
        Get recent feedback entries

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of feedback records
        """
        cursor = self.conn.execute(
            'SELECT * FROM feedback ORDER BY timestamp DESC LIMIT ?',
            (limit,)
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_feedback_stats(self):
        """
        Get statistics about feedback

        Returns:
            Dict with stats
        """
        cursor = self.conn.execute('''
            SELECT
                COUNT(*) as total_feedback,
                AVG(rating) as avg_rating,
                COUNT(DISTINCT report_id) as num_reports
            FROM feedback
        ''')
        row = cursor.fetchone()
        return dict(row) if row else {}

    def delete_feedback(self, report_id, student_id):
        """
        Delete feedback entry

        Args:
            report_id: Report identifier
            student_id: Student ID

        Returns:
            True if successful
        """
        try:
            self.conn.execute(
                'DELETE FROM feedback WHERE report_id = ? AND student_id = ?',
                (report_id, student_id)
            )
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            return False

    def export_as_csv(self, output_path=None):
        """
        Export all feedback as CSV

        Args:
            output_path: Path to save CSV (if None, returns string)

        Returns:
            CSV content as string or path
        """
        import pandas as pd

        cursor = self.conn.execute(
            'SELECT * FROM feedback ORDER BY timestamp DESC')
        rows = [dict(row) for row in cursor.fetchall()]
        df = pd.DataFrame(rows)

        if output_path:
            df.to_csv(output_path, index=False)
            return output_path
        else:
            return df.to_csv(index=False)

    def close(self):
        """Close database connection"""
        self.conn.close()

    def __del__(self):
        """Ensure database is closed on cleanup"""
        try:
            self.close()
        except:
            pass
