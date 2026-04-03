"""Output generation — event streams, book records, writers, and reports."""

from .audit import AuditReport, ArchetypeSurvival, format_audit_report, run_audit, write_audit_report
from .book_record import BookRecord, book_record_from_instance
from .book_writer import BookWriter, BookWriterConfig
from .event_stream import EventStreamGenerator
from .lookup_writer import LookupTableWriter

__all__ = [
    "ArchetypeSurvival",
    "AuditReport",
    "BookRecord",
    "BookWriter",
    "BookWriterConfig",
    "EventStreamGenerator",
    "LookupTableWriter",
    "book_record_from_instance",
    "format_audit_report",
    "run_audit",
    "write_audit_report",
]
