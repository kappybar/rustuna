use std::collections::VecDeque;

use crate::{Error, ErrorKind, Result};

pub trait TrialQueue: Send + Sync {
    fn enqueue(&mut self, trial_id: u32) -> Result<()>;
    fn dequeue(&mut self) -> Result<u32>;
}

#[derive(Default)]
pub struct InMemoryTrialQueue {
    queue: VecDeque<u32>,
}

impl InMemoryTrialQueue {
    pub fn new() -> Self {
        Self {
            queue: VecDeque::new(),
        }
    }
}

impl TrialQueue for InMemoryTrialQueue {
    fn enqueue(&mut self, trial_id: u32) -> Result<()> {
        self.queue.push_back(trial_id);
        Ok(())
    }

    fn dequeue(&mut self) -> Result<u32> {
        self.queue
            .pop_back()
            .ok_or_else(|| Error::new(ErrorKind::TrialQueueEmpty))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enqueue_and_dequeue() {
        let mut queue = InMemoryTrialQueue::new();

        assert!(queue.enqueue(1).is_ok());
        assert!(queue.enqueue(2).is_ok());
        assert!(queue.enqueue(3).is_ok());

        assert_eq!(queue.dequeue().unwrap(), 3);
        assert_eq!(queue.dequeue().unwrap(), 2);
        assert_eq!(queue.dequeue().unwrap(), 1);
    }

    #[test]
    fn test_dequeue_empty_queue() {
        let mut queue = InMemoryTrialQueue::new();
        assert!(queue.dequeue().is_err());
    }
}
