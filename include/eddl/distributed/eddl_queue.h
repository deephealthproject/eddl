/*
 * EDDL Library - European Distributed Deep Learning Library.
 * Version: x.y
 * copyright (c) 2020, Universitat Politècnica de València (UPV), PRHLT Research Centre
 * Date: July 2020
 * Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
 * All rights reserved
 */

#ifndef __EDDL_QUEUE_H__
#define __EDDL_QUEUE_H__ 1

#include <queue>
#include <mutex>
#include <condition_variable>

#include <eddl/distributed/eddl_message.h>

namespace eddl {

class eddl_queue
{
public:
    eddl_queue() {}

    ~eddl_queue()
    {
        clear();
    }

    void clear()
    {
        // Critical region starts
        std::unique_lock<std::mutex> lck(mutex_queue);

        while (! q.empty()) {
            eddl_message *m = q.front();
            q.pop();
            delete m;
        }
        // Critical region ends
    }

    void push(eddl_message * message)
    {
        // Critical region starts
        std::unique_lock<std::mutex> lck(mutex_queue);
        q.push(message);
        cond_var.notify_one();
        // Critical region ends
    }

    void push_front(eddl_message * message)
    {
        // Critical region starts
        std::unique_lock<std::mutex> lck(mutex_queue);
        q.push(message);
        for (auto i = q.size(); i > 0; i--) {
            q.push(q.front());
            q.pop();
        }
        cond_var.notify_one();
        // Critical region ends
    }

    eddl_message * front()
    {
        eddl_message * message = nullptr;
        // Critical region starts
        std::unique_lock<std::mutex> lck(mutex_queue);

        if ( q.empty() ) cond_var.wait(lck);

        if ( ! q.empty() ) {
            message = q.front();
        }

        return message;
        // Critical region ends
    }

    eddl_message * pop()
    {
        eddl_message * message = nullptr;
        // Critical region starts
        std::unique_lock<std::mutex> lck(mutex_queue);

        if ( q.empty() ) cond_var.wait(lck);

        if ( ! q.empty() ) {
            message = q.front();
            q.pop();
        }

        return message;
        // Critical region ends
    }

    size_t size()
    {
        // Critical region starts
        std::unique_lock<std::mutex> lck(mutex_queue);
        return q.size();
        // Critical region ends
    }

    bool empty()
    {
        // Critical region starts
        std::unique_lock<std::mutex> lck(mutex_queue);
        return q.empty();
        // Critical region ends
    }

private:
    std::queue<eddl_message *>  q; // the actual queue
    std::mutex                  mutex_queue;
    std::condition_variable     cond_var;
};

};

#endif // __EDDL_QUEUE_H__
